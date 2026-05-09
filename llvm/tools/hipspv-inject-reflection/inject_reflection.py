#!/usr/bin/env python3
"""Inject NonSemantic.ClspvReflection.5 + spec-constant LocalSizeId into a
Vulkan SPIR-V emitted by HIPSPV's bridging pass.

If a reflection JSON is supplied, it is used as-is. Otherwise the script
auto-discovers kernel + argument layout from the SPV's OpEntryPoint /
DescriptorSet / Binding decorations and from the PushConstant struct's
OpMemberDecorate Offset chain.

Usage: inject_reflection.py <in.spv> <out.spv> [<reflect.json>]
"""
import json
import struct
import sys

# SPIR-V opcodes
OP_EXTENSION = 10
OP_EXT_INST_IMPORT = 11
OP_STRING = 7
OP_EXT_INST = 12
OP_TYPE_INT = 21
OP_TYPE_VOID = 19
OP_TYPE_VECTOR = 23
OP_TYPE_POINTER = 32
OP_CONSTANT = 43
OP_SPEC_CONSTANT = 50
OP_SPEC_CONSTANT_COMPOSITE = 51
OP_VARIABLE = 59
OP_LOAD = 61
OP_ACCESS_CHAIN = 65
OP_DECORATE = 71
OP_NAME = 5
OP_EXECUTION_MODE = 16
OP_EXECUTION_MODE_ID = 331

DECORATION_SPEC_ID = 1
DECORATION_BUILTIN = 11
BUILTIN_WORKGROUP_SIZE = 25
EXECUTION_MODE_LOCAL_SIZE = 17
EXECUTION_MODE_LOCAL_SIZE_ID = 38

# NonSemantic.ClspvReflection.5 instruction numbers
KERNEL = 1
ARG_STORAGE_BUFFER = 3
ARG_POD_PUSH_CONSTANT = 7
PROGRAM_SCOPE_VARIABLES_STORAGE_BUFFER = 28

def read_spv(path):
    with open(path, 'rb') as f:
        data = f.read()
    words = struct.unpack(f'<{len(data)//4}I', data)
    return list(words)

def write_spv(path, words):
    with open(path, 'wb') as f:
        f.write(struct.pack(f'<{len(words)}I', *words))

def encode_string(s):
    """SPIR-V string: null-terminated, padded to multiple of 4 bytes,
    packed into little-endian words."""
    b = s.encode('utf-8') + b'\0'
    while len(b) % 4 != 0:
        b += b'\0'
    return list(struct.unpack(f'<{len(b)//4}I', b))

def make_instr(opcode, operands):
    word_count = 1 + len(operands)
    return [(word_count << 16) | opcode] + list(operands)

def parse_instructions(words):
    """Yield (offset, opcode, word_count, operands) over the instruction stream."""
    i = 5  # skip header
    while i < len(words):
        w = words[i]
        wc = (w >> 16) & 0xFFFF
        op = w & 0xFFFF
        operands = words[i+1:i+wc]
        yield i, op, wc, operands
        i += wc

def find_function_id(words, name):
    """Find the function's result-id by scanning OpName + OpFunction. We look
    for OpName <id> "name", and trust that the named id is a function."""
    for off, op, wc, ops in parse_instructions(words):
        if op == OP_NAME and len(ops) >= 2:
            target = ops[0]
            # decode null-terminated name
            name_bytes = struct.pack(f'<{len(ops)-1}I', *ops[1:])
            decoded = name_bytes.split(b'\0', 1)[0].decode('utf-8', errors='replace')
            if decoded == name:
                return target
    return None

def find_or_make_int_type_and_const(words, header_bound, value):
    """Locate uint32 type id and an OpConstant for `value`. If not present, return
    (None, None) — caller will insert. For simplicity here, we always insert new."""
    return None, None


def autodiscover_kernels(words):
    """Walk the SPIR-V module to recover kernel and arg layout from existing
    decorations. Returns a list shaped like the JSON input.

    Strategy:
      - OpEntryPoint gives kernel name + entry function id + interface vars.
      - Each interface var is either a Vulkan builtin (Input storage class)
        or a buffer global (StorageBuffer/PushConstant).
      - StorageBuffer vars carry DescriptorSet+Binding decorations.
      - The PushConstant variable's pointee struct has OpMemberDecorate Offset
        + we know the struct field types from OpTypeStruct + OpTypeInt etc.
      - Source-arg ordinals are assigned: pointer args first (in binding-id
        order), then scalar args (in struct-field order).
    """
    OP_ENTRY_POINT = 15
    OP_TYPE_STRUCT = 30
    OP_TYPE_POINTER = 32
    OP_TYPE_INT = 21
    OP_TYPE_FLOAT = 22
    OP_TYPE_VECTOR_LOCAL = 23
    OP_TYPE_ARRAY = 28
    OP_TYPE_MATRIX = 24
    OP_CONSTANT_LOCAL = 43

    DECORATION_DESCRIPTOR_SET = 34
    DECORATION_BINDING = 33
    DECORATION_OFFSET = 35

    STORAGE_CLASS_PUSH_CONSTANT = 9
    STORAGE_CLASS_STORAGE_BUFFER = 12

    var_set = {}      # var_id -> descriptor set
    var_binding = {}  # var_id -> binding
    var_storage = {}  # var_id -> storage class
    var_pointee = {}  # var_id -> type id of the pointee
    var_name_map = {} # var_id -> OpName string
    pointer_to_type = {}  # ptr_type_id -> (storage_class, pointee_type_id)
    type_member_offsets = {}  # struct_type_id -> {idx: offset}
    type_int_width = {}  # int_type_id -> bit width
    type_float_width = {}  # float_type_id -> bit width
    type_struct_members = {}  # struct_type_id -> list of member type ids
    type_vector_info = {}  # vector_type_id -> (component_type_id, count)
    type_array_info = {}   # array_type_id -> (element_type_id, length_const_id)
    type_matrix_info = {}  # matrix_type_id -> (column_type_id, column_count)
    int_const_value = {}   # const_id -> integer value (for OpConstant of uint/int)

    entry_points = []  # list of (fn_id, name, [iface_var_ids])
    OP_NAME_OPCODE = 5

    for off, op, wc, ops in parse_instructions(words):
        if op == OP_ENTRY_POINT and len(ops) >= 3:
            # ops[0]=execution model, ops[1]=fn id, ops[2:]=name string + interface
            fn_id = ops[1]
            j = 2
            name_bytes = b''
            while j < len(ops):
                w = ops[j]
                name_bytes += struct.pack('<I', w)
                j += 1
                if any(((w >> (8*k)) & 0xff) == 0 for k in range(4)):
                    break
            name = name_bytes.split(b'\0', 1)[0].decode('utf-8', errors='replace')
            iface = list(ops[j:])
            entry_points.append((fn_id, name, iface))
        elif op == OP_NAME_OPCODE and len(ops) >= 2:
            tgt = ops[0]
            nb = struct.pack(f'<{len(ops)-1}I', *ops[1:])
            var_name_map[tgt] = nb.split(b'\0', 1)[0].decode('utf-8', errors='replace')
        elif op == OP_DECORATE and len(ops) >= 3:
            tgt = ops[0]
            if ops[1] == DECORATION_DESCRIPTOR_SET:
                var_set[tgt] = ops[2]
            elif ops[1] == DECORATION_BINDING:
                var_binding[tgt] = ops[2]
        elif op == 72 and len(ops) >= 4 and ops[2] == DECORATION_OFFSET:
            # OpMemberDecorate <struct_id> <member> Offset <bytes>
            type_member_offsets.setdefault(ops[0], {})[ops[1]] = ops[3]
        elif op == OP_TYPE_POINTER and len(ops) >= 3:
            pointer_to_type[ops[0]] = (ops[1], ops[2])  # storage, pointee
        elif op == OP_VARIABLE and len(ops) >= 3:
            ptr_ty = ops[0]
            var_id = ops[1]
            sc = ops[2]
            var_storage[var_id] = sc
            if ptr_ty in pointer_to_type:
                _, pointee = pointer_to_type[ptr_ty]
                var_pointee[var_id] = pointee
        elif op == OP_TYPE_INT and len(ops) >= 2:
            type_int_width[ops[0]] = ops[1]
        elif op == OP_TYPE_FLOAT and len(ops) >= 2:
            type_float_width[ops[0]] = ops[1]
        elif op == OP_TYPE_STRUCT and len(ops) >= 1:
            type_struct_members[ops[0]] = list(ops[1:])
        elif op == OP_TYPE_VECTOR_LOCAL and len(ops) >= 3:
            type_vector_info[ops[0]] = (ops[1], ops[2])
        elif op == OP_TYPE_ARRAY and len(ops) >= 3:
            type_array_info[ops[0]] = (ops[1], ops[2])
        elif op == OP_TYPE_MATRIX and len(ops) >= 3:
            type_matrix_info[ops[0]] = (ops[1], ops[2])
        elif op == OP_CONSTANT_LOCAL and len(ops) >= 3:
            # OpConstant <result_type> <result_id> <literal>
            # only record when result_type is an integer type we know about
            if ops[0] in type_int_width:
                int_const_value[ops[1]] = ops[2]

    def field_size_bytes(type_id, _seen=None):
        # Compute the byte size of a SPIR-V type. Handles scalars, vectors,
        # arrays, matrices, and structs.
        #
        # For structs, recursively sums member sizes. Single-member structs
        # (e.g. `struct.__half` wrapping a 16-bit int, or HIP_vector_type
        # wrapping a `<4 x float>`) collapse naturally — the sum of one
        # member equals that member's size.
        if _seen is None:
            _seen = set()
        if type_id in _seen:
            return 4
        _seen.add(type_id)
        if type_id in type_int_width:
            return (type_int_width[type_id] + 7) // 8
        if type_id in type_float_width:
            return (type_float_width[type_id] + 7) // 8
        if type_id in type_vector_info:
            comp_ty, count = type_vector_info[type_id]
            return field_size_bytes(comp_ty, _seen) * count
        if type_id in type_array_info:
            elem_ty, length_const = type_array_info[type_id]
            length = int_const_value.get(length_const, 1)
            return field_size_bytes(elem_ty, _seen) * length
        if type_id in type_matrix_info:
            col_ty, col_count = type_matrix_info[type_id]
            return field_size_bytes(col_ty, _seen) * col_count
        if type_id in type_struct_members:
            members = type_struct_members[type_id]
            return sum(field_size_bytes(m, _seen.copy()) for m in members)
        return 4

    kernels = []
    # Phase G4: device globals live at descriptor set 1; kernel args live at
    # descriptor set 0. Distinguish them so the auto-discovered reflection
    # only emits ArgumentStorageBuffer entries for actual kernel-arg buffers,
    # not for `__device__` globals (which the chipStar runtime needs to track
    # separately for hipMemcpyToSymbol).
    # Per-kernel EP interface filter: chipstarvulkan emits a single OpEntryPoint
    # interface that lists ALL module-level resources (every kernel's SBs and
    # PCs). The reflection must be emitted per-kernel — only the resources
    # the kernel actually consumes. Each resource var carries an OpName whose
    # leading token equals the mangled kernel name (e.g.
    # `_Z5haxpy6__halfPKS_PS_.0`, `.1`, `.pcs`, or
    # `.hipspv.pcs._Z5haxpy6__halfPKS_PS_.1`). Use that prefix to filter.
    def belongs_to_kernel(var_id, kernel_mangled):
        nm = var_name_map.get(var_id, "")
        if not nm:
            return False
        # Phase I3: device-global descriptors emitted by the bridging pass
        # carry names of the form `__hipspv_dg_<symbol>[__sz_<n>__init_<hex>]`
        # rather than the per-kernel `<kernel>.<idx>` convention. They are
        # always part of any kernel that references the symbol — the bridging
        # pass scopes them to that kernel by adding a hidden ptr arg, so the
        # OpEntryPoint interface only lists the resource for kernels that
        # actually use it. Treat any `__hipspv_dg_` resource appearing in
        # this kernel's interface as belonging to the kernel.
        if nm.startswith("__hipspv_dg_"):
            return True
        return kernel_mangled in nm

    for fn_id, name, iface in entry_points:
        sb_vars = [v for v in iface
                   if var_storage.get(v) == STORAGE_CLASS_STORAGE_BUFFER
                   and var_set.get(v, 0) == 0
                   and belongs_to_kernel(v, name)]
        sb_vars.sort(key=lambda v: var_binding.get(v, 0))
        args = []
        for ord_, v in enumerate(sb_vars):
            args.append({
                "kind": "storage_buffer",
                "ord": ord_,
                "set": var_set.get(v, 0),
                "binding": var_binding.get(v, ord_),
            })
        next_ord = len(args)

        # Push constant args: walk the PushConstant variable's struct fields.
        # Only consider the PC variable that belongs to this kernel.
        pc_vars = [v for v in iface
                   if var_storage.get(v) == STORAGE_CLASS_PUSH_CONSTANT
                   and belongs_to_kernel(v, name)]
        for v in pc_vars:
            struct_id = var_pointee.get(v)
            if struct_id is None or struct_id not in type_struct_members:
                continue
            members = type_struct_members[struct_id]
            offsets = type_member_offsets.get(struct_id, {})
            for fi, mty in enumerate(members):
                args.append({
                    "kind": "pod_push_constant",
                    "ord": next_ord,
                    "offset": offsets.get(fi, 0),
                    "size": field_size_bytes(mty),
                })
                next_ord += 1

        kernels.append({"name": name, "args": args})

    # Device-global discovery: every StorageBuffer var at descriptor set 1 is
    # a `__device__` global lowered by the bridging pass. The OpName follows
    # the pattern `<symbol>.<kernel>`; extract the symbol prefix.
    OP_NAME_OPCODE = 5
    var_name = {}
    for off, op, wc, ops in parse_instructions(words):
        if op == OP_NAME_OPCODE and len(ops) >= 2:
            tgt = ops[0]
            name_bytes = struct.pack(f'<{len(ops)-1}I', *ops[1:])
            decoded = name_bytes.split(b'\0', 1)[0].decode('utf-8', errors='replace')
            var_name[tgt] = decoded

    device_globals = []
    seen_bindings = set()
    for v, sc in var_storage.items():
        if sc != STORAGE_CLASS_STORAGE_BUFFER:
            continue
        if var_set.get(v, 0) != 1:
            continue
        bind = var_binding.get(v, 0)
        if bind in seen_bindings:
            continue
        seen_bindings.add(bind)
        full_name = var_name.get(v, "")
        # Strip trailing ".<kernel_name>" suffix that the bridging pass
        # appended for OpName uniqueness.
        sym = full_name.split('.', 1)[0] if full_name else f"dg_{bind}"
        device_globals.append({
            "name": sym,
            "set": 1,
            "binding": bind,
        })

    return {"kernels": kernels, "device_globals": device_globals}

def _rewrite_localsize_only(words, header_bound, exec_mode_off,
                            entrypoint_fn_id, uint_id):
    """Convert OpExecutionMode LocalSize x y z → OpExecutionModeId LocalSizeId
    with three SpecId-decorated spec constants and a SpecConstantComposite
    BuiltIn WorkgroupSize. Used when the kernel does not reference
    get_local_size() (so no BuiltIn WorkgroupSize OpVariable was emitted)
    but still has a fixed OpExecutionMode LocalSize that must be made
    overridable for Vulkan dispatch.
    """
    bound = header_bound

    # Locate / synthesise needed type ids.
    v3uint_id = None
    for _, op, _, ops in parse_instructions(words):
        if op == OP_TYPE_INT and len(ops) >= 3 and ops[1] == 32 and ops[2] == 0:
            uint_id = ops[0]
        elif op == OP_TYPE_VECTOR and uint_id is not None and \
                len(ops) >= 3 and ops[1] == uint_id and ops[2] == 3:
            v3uint_id = ops[0]

    def newid():
        nonlocal bound
        bound += 1
        return bound

    new_types = []
    if uint_id is None:
        uint_id = newid()
        new_types += make_instr(OP_TYPE_INT, [uint_id, 32, 0])
    if v3uint_id is None:
        v3uint_id = newid()
        new_types += make_instr(OP_TYPE_VECTOR, [v3uint_id, uint_id, 3])

    sc_x = newid(); sc_y = newid(); sc_z = newid(); sc_wg = newid()

    new_decorates = (
        make_instr(OP_DECORATE, [sc_x, DECORATION_SPEC_ID, 0]) +
        make_instr(OP_DECORATE, [sc_y, DECORATION_SPEC_ID, 1]) +
        make_instr(OP_DECORATE, [sc_z, DECORATION_SPEC_ID, 2]) +
        make_instr(OP_DECORATE, [sc_wg, DECORATION_BUILTIN, BUILTIN_WORKGROUP_SIZE])
    )
    new_constants = (
        make_instr(OP_SPEC_CONSTANT, [uint_id, sc_x, 1]) +
        make_instr(OP_SPEC_CONSTANT, [uint_id, sc_y, 1]) +
        make_instr(OP_SPEC_CONSTANT, [uint_id, sc_z, 1]) +
        make_instr(OP_SPEC_CONSTANT_COMPOSITE, [v3uint_id, sc_wg, sc_x, sc_y, sc_z])
    )

    # SPIR-V layout: decorations MUST come before the type-declaration
    # section. For trivial kernels (e.g. log2f) with no existing OpDecorate,
    # inject our new decorates right before the first OpType*. Likewise
    # types/constants must precede OpFunction.
    #
    # IMPORTANT ordering constraint: types and global variables are
    # interleaved within section 9 of the SPIR-V logical layout. The
    # original module may declare OpVariable instances BEFORE the
    # OpTypeVector for v3uint (e.g. PushConstant variables that don't
    # depend on v3uint precede the WorkgroupId/LocalInvocationId variables
    # whose pointer types reference v3uint). We must insert
    # `new_constants` (which references v3uint_id via OpSpecConstantComposite)
    # only AFTER v3uint has been emitted, otherwise the SPIR-V validator
    # rejects the module with "ID has not been defined" forward-reference
    # errors. The safest landmark is OpFunction, which by spec follows the
    # entire type/global-var/constant section.
    OP_FUNCTION = 0x36
    # Detect whether v3uint already exists in the original module by
    # checking whether new_types is empty (set above): if non-empty we
    # are synthesising it and will emit it ourselves; if empty, v3uint
    # is declared somewhere in the original word stream and we must wait
    # until that declaration has been emitted before inserting constants.
    v3uint_emitted = bool(new_types)  # if synthesising, becomes True after we emit new_types
    out = list(words[:5])
    i = 5
    inserted_decorates = False
    inserted_types = False
    inserted_constants = False
    while i < len(words):
        w = words[i]
        wc = (w >> 16) & 0xFFFF
        op = w & 0xFFFF
        instr = words[i:i + wc]

        # Replace the fixed-LocalSize execution mode with LocalSizeId.
        if i == exec_mode_off:
            out.extend(make_instr(OP_EXECUTION_MODE_ID,
                                  [entrypoint_fn_id, EXECUTION_MODE_LOCAL_SIZE_ID,
                                   sc_x, sc_y, sc_z]))
            i += wc; continue

        # Inject SpecId / BuiltIn decorations right before the first
        # OpDecorate, OR right before the first OpType*/OpConstant if the
        # module has no decorates (signals end of annotation section).
        if not inserted_decorates:
            if op == OP_DECORATE:
                out.extend(instr)
                out.extend(new_decorates)
                inserted_decorates = True
                i += wc; continue
            # OpType* opcodes are in range 0x13..0x2D in SPIR-V spec.
            # OpConstant opcodes start at 0x2A. Either signals end of
            # annotation section.
            if 0x13 <= op <= 0x36:
                out.extend(new_decorates)
                inserted_decorates = True
                # fall through to emit the type/function instruction

        # Inject synthesised types before the first OpVariable, or before
        # OpFunction if the module has no OpVariable (trivial kernel).
        # If we synthesise v3uint here, mark it as emitted so constants can
        # follow. If we did NOT synthesise (v3uint already in module), do
        # NOT mark v3uint_emitted yet — wait for the OpTypeVector to be
        # written through the original word stream.
        if not inserted_types and (op == OP_VARIABLE or op == OP_FUNCTION):
            if new_types:
                out.extend(new_types)
                v3uint_emitted = True
            inserted_types = True

        # Inject our spec-constants (incl. SpecConstantComposite that
        # references v3uint_id) only AFTER v3uint has been declared, and
        # before OpFunction at the latest.
        if not inserted_constants and v3uint_emitted and op == OP_FUNCTION:
            out.extend(new_constants)
            inserted_constants = True

        out.extend(instr)

        # If this instruction was the v3uint type declaration, mark it
        # so the next OpFunction will trigger constants injection.
        if (not v3uint_emitted and op == OP_TYPE_VECTOR and
                len(instr) >= 4 and instr[1] == v3uint_id):
            v3uint_emitted = True

        i += wc

    if not inserted_decorates:
        out.extend(new_decorates)
    if not inserted_types and new_types:
        out.extend(new_types)
    if not inserted_constants:
        out.extend(new_constants)

    out[3] = bound + 1
    return out, bound


def rewrite_workgroup_size(words, header_bound):
    """In-place rewrite of `BuiltIn WorkgroupSize` OpVariable into the
    canonical spec-constant pattern Vulkan requires:
        OpDecorate %sc_x SpecId 0  (similarly y/z = SpecId 1/2)
        OpDecorate %sc_wg BuiltIn WorkgroupSize
        %sc_x = OpSpecConstant %uint 1
        %sc_y = OpSpecConstant %uint 1
        %sc_z = OpSpecConstant %uint 1
        %sc_wg = OpSpecConstantComposite %v3uint %sc_x %sc_y %sc_z
    Plus rewrites loads of the WG-size variable to use the composite, and
    swaps OpExecutionMode LocalSize → OpExecutionModeId LocalSizeId.
    Returns (modified_words, new_bound).
    """
    bound = header_bound

    # Find the WorkgroupSize variable id and its decoration instruction offset.
    wg_var_id = None
    wg_decorate_off = None
    wg_var_off = None
    v3uint_id = None
    uint_id = None
    ptr_input_v3uint_id = None
    exec_mode_off = None
    entrypoint_fn_id = None

    for off, op, wc, ops in parse_instructions(words):
        # OpDecorate <target> <decoration_kind> <decoration_args...>
        if (op == OP_DECORATE and len(ops) >= 3 and
                ops[1] == DECORATION_BUILTIN and ops[2] == BUILTIN_WORKGROUP_SIZE):
            wg_var_id = ops[0]
            wg_decorate_off = off
        elif op == OP_TYPE_INT and len(ops) >= 3 and ops[1] == 32 and ops[2] == 0:
            uint_id = ops[0]
        elif op == OP_EXECUTION_MODE and len(ops) >= 2 and \
                ops[1] == EXECUTION_MODE_LOCAL_SIZE:
            exec_mode_off = off
            entrypoint_fn_id = ops[0]

    if wg_var_id is None:
        # No BuiltIn WorkgroupSize variable — but if there's still an
        # OpExecutionMode LocalSize x y z, Vulkan needs it converted to
        # OpExecutionModeId LocalSizeId with spec constants so the host
        # can override the workgroup dimensions at dispatch time.
        if exec_mode_off is None:
            return words, bound  # truly nothing to do
        return _rewrite_localsize_only(words, bound, exec_mode_off,
                                       entrypoint_fn_id, uint_id)

    # Find the OpVariable for that id and the v3uint type chain.
    for off, op, wc, ops in parse_instructions(words):
        if op == OP_VARIABLE and len(ops) >= 2 and ops[1] == wg_var_id:
            wg_var_off = off
            ptr_input_v3uint_id = ops[0]
        elif op == OP_TYPE_VECTOR and uint_id is not None and \
                len(ops) >= 3 and ops[1] == uint_id and ops[2] == 3:
            v3uint_id = ops[0]

    if uint_id is None or v3uint_id is None or wg_var_off is None:
        return words, bound

    def newid():
        nonlocal bound
        bound += 1
        return bound

    sc_x = newid(); sc_y = newid(); sc_z = newid(); sc_wg = newid()

    # New instruction blobs.
    new_decorates = (
        make_instr(OP_DECORATE, [sc_x, DECORATION_SPEC_ID, 0]) +
        make_instr(OP_DECORATE, [sc_y, DECORATION_SPEC_ID, 1]) +
        make_instr(OP_DECORATE, [sc_z, DECORATION_SPEC_ID, 2]) +
        make_instr(OP_DECORATE, [sc_wg, DECORATION_BUILTIN, BUILTIN_WORKGROUP_SIZE])
    )
    new_constants = (
        make_instr(OP_SPEC_CONSTANT, [uint_id, sc_x, 1]) +
        make_instr(OP_SPEC_CONSTANT, [uint_id, sc_y, 1]) +
        make_instr(OP_SPEC_CONSTANT, [uint_id, sc_z, 1]) +
        make_instr(OP_SPEC_CONSTANT_COMPOSITE, [v3uint_id, sc_wg, sc_x, sc_y, sc_z])
    )

    # Rewrite the module:
    #   - delete original BuiltIn WorkgroupSize decoration
    #   - delete OpVariable for wg_var_id
    #   - replace OpLoad <v3uint> %wg_var → use sc_wg directly downstream
    #   - replace OpExecutionMode LocalSize → OpExecutionModeId LocalSizeId sc_x sc_y sc_z
    #   - inject new decorations + constants

    # First pass: collect load result-ids that read the wg_var, so we can RAUW.
    load_id_to_replace = {}  # old_load_result_id -> sc_wg
    for off, op, wc, ops in parse_instructions(words):
        if op == OP_LOAD and len(ops) >= 3 and ops[2] == wg_var_id:
            # ops: [result_type, result_id, pointer (= wg_var_id), [memory ops...]]
            load_id_to_replace[ops[1]] = sc_wg

    # Second pass: rebuild the word stream.
    OP_ENTRY_POINT = 15
    out = list(words[:5])  # header
    i = 5
    inserted_decorates = False
    inserted_constants = False
    while i < len(words):
        w = words[i]
        wc = (w >> 16) & 0xFFFF
        op = w & 0xFFFF
        instr = words[i:i+wc]

        # Drop original BuiltIn WorkgroupSize decoration on the variable.
        if i == wg_decorate_off:
            i += wc; continue
        # Drop the OpVariable for the wgsize variable.
        if i == wg_var_off:
            i += wc; continue
        # Drop OpLoad on wg_var (its result-id is RAUW'd at downstream uses).
        if op == OP_LOAD and len(instr) >= 4 and instr[3] == wg_var_id:
            i += wc; continue
        # Drop OpName referring to wg_var.
        if op == OP_NAME and len(instr) >= 2 and instr[1] == wg_var_id:
            i += wc; continue
        # OpEntryPoint contains a list of interface variable IDs at the tail —
        # strip wg_var_id from that list.
        if op == OP_ENTRY_POINT:
            new_instr = [instr[0], instr[1], instr[2]]  # opcode-word, exec model, fn id
            # Then a literal name string (variable length up to first 0 in last byte
            # of a word). We'll just copy until we find ids that look like the
            # interface list. Find the end of the name: word with a zero byte.
            j = 3
            while j < len(instr):
                ww = instr[j]
                new_instr.append(ww)
                # Detect string-terminator word (any zero byte).
                if any(((ww >> (8*k)) & 0xff) == 0 for k in range(4)):
                    j += 1; break
                j += 1
            # Remaining words are interface variable IDs; filter out wg_var_id.
            for k in range(j, len(instr)):
                if instr[k] == wg_var_id:
                    continue
                new_instr.append(instr[k])
            # Fix word count in the leading word.
            new_wc = len(new_instr)
            new_instr[0] = (new_wc << 16) | OP_ENTRY_POINT
            out.extend(new_instr)
            i += wc; continue
        # Replace OpExecutionMode LocalSize with OpExecutionModeId LocalSizeId.
        if i == exec_mode_off:
            new_em = make_instr(OP_EXECUTION_MODE_ID,
                                [entrypoint_fn_id, EXECUTION_MODE_LOCAL_SIZE_ID,
                                 sc_x, sc_y, sc_z])
            out.extend(new_em)
            i += wc; continue
        # Inject our new decorations right before the first OpDecorate.
        if not inserted_decorates and op == OP_DECORATE:
            out.extend(instr)
            out.extend(new_decorates)
            inserted_decorates = True
            i += wc; continue
        # Inject new constants only after v3uint has been declared and
        # before OpFunction. PushConstant or other OpVariables can
        # legitimately precede OpTypeVector v3uint in the original module,
        # so emitting new_constants at the first OpVariable would create a
        # forward reference to %v3uint inside OpSpecConstantComposite.
        if not inserted_constants and op == 0x36:  # OpFunction
            out.extend(new_constants)
            inserted_constants = True
            # fall through to emit current instruction
        # RAUW any operand that was a load-of-wgsize result-id.
        # CAUTION: several opcodes carry literal (non-ID) operands whose
        # numeric value can coincidentally equal a load result-id and must
        # NOT be remapped. The most painful case is OpConstant: a kernel-
        # name string lowered to [N x i8] becomes OpConstant %uchar 'S'(83),
        # 'p'(112), etc., and 83/112 happen to be the load-result-ids of
        # the WorkgroupSize OpLoad we are RAUW-ing here. Blind RAUW would
        # rewrite the byte literal 83 → sc_wg id, producing OpConstant
        # %uchar 300 which the SPIR-V validator rejects (high-order bits
        # set in an 8-bit literal).
        # Skip RAUW entirely on opcodes whose word stream contains literals.
        LITERAL_BEARING_OPS = {
            7,    # OpString (packed UTF-8 bytes)
            5,    # OpName (id, then packed string)
            6,    # OpMemberName (id, member-idx literal, packed string)
            8,    # OpLine (file-id, line literal, col literal)
            16,   # OpExecutionMode (entry-id, mode literal, mode args literal)
            17,   # OpCapability (capability literal)
            10,   # OpExtension (string)
            11,   # OpExtInstImport (id, string)
            14,   # OpMemoryModel (literals)
            15,   # OpEntryPoint (mode literal + name string + ids)
            43,   # OpConstant (type-id, result-id, literal bit pattern)
            50,   # OpSpecConstant (same)
            71,   # OpDecorate (id, deco-kind literal, deco-args literal)
            72,   # OpMemberDecorate (id, member-idx literal, deco literals)
            73,   # OpGroupDecorate
            74,   # OpGroupMemberDecorate
            75,   # OpDecorationGroup
            331,  # OpExecutionModeId — first arg after fn-id IS literal mode
        }
        if load_id_to_replace and len(instr) > 1 and op not in LITERAL_BEARING_OPS:
            instr = list(instr)
            for j in range(1, len(instr)):
                if instr[j] in load_id_to_replace:
                    instr[j] = load_id_to_replace[instr[j]]
        out.extend(instr)
        i += wc

    # Update header bound.
    out[3] = bound + 1
    return out, bound

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    in_path, out_path = sys.argv[1:3]
    reflect_json = sys.argv[3] if len(sys.argv) >= 4 else None

    words = read_spv(in_path)
    bound = words[3]
    if reflect_json:
        info = json.loads(open(reflect_json).read())
    else:
        info = autodiscover_kernels(words)
        if not info["kernels"]:
            print("warning: autodiscover found no kernels; emitting unchanged module",
                  file=sys.stderr)
            write_spv(out_path, words)
            return

    # Allocate new IDs lazily.
    def newid():
        nonlocal bound
        bound += 1
        return bound

    # Step 1: build OpExtension + OpExtInstImport for the reflection set.
    ext_decl = make_instr(OP_EXTENSION,
                          encode_string("SPV_KHR_non_semantic_info"))
    import_id = newid()
    name_words = encode_string("NonSemantic.ClspvReflection.5")
    ext_import = make_instr(OP_EXT_INST_IMPORT, [import_id] + name_words)

    # Find a void type; if absent, synthesise.
    void_id = None
    uint_id = None
    for _, op, _, ops in parse_instructions(words):
        if op == OP_TYPE_VOID:
            void_id = ops[0]
        if op == OP_TYPE_INT and len(ops) >= 3 and ops[1] == 32 and ops[2] == 0:
            uint_id = ops[0]
        if void_id and uint_id:
            break

    inserts_after_types = []
    if void_id is None:
        void_id = newid()
        inserts_after_types += make_instr(OP_TYPE_VOID, [void_id])
    if uint_id is None:
        uint_id = newid()
        inserts_after_types += make_instr(OP_TYPE_INT, [uint_id, 32, 0])

    # Step 2: build OpConstant uint32 cache so we don't keep re-emitting.
    const_cache = {}
    const_decls = []
    def uint_const(v):
        if v not in const_cache:
            cid = newid()
            const_cache[v] = cid
            const_decls.extend(make_instr(OP_CONSTANT, [uint_id, cid, v]))
        return const_cache[v]

    # Step 3: emit reflection per kernel.
    # ExtInstKernel signature (per clspv SPIRVProducerPass):
    #   void_id result_id ImportSet KERNEL fn_id name_str_id num_args_const flags_const attrs_str_id
    string_block = []
    refl_block = []
    # One empty attributes OpString reused across kernels.
    attrs_id = newid()
    string_block.extend(make_instr(OP_STRING, [attrs_id] + encode_string("")))

    for k in info["kernels"]:
        fn_id = find_function_id(words, k["name"])
        if fn_id is None:
            print(f"Could not locate function id for kernel {k['name']!r}", file=sys.stderr)
            sys.exit(2)
        name_id = newid()
        string_block.extend(make_instr(OP_STRING, [name_id] + encode_string(k["name"])))
        kernel_result = newid()
        num_args_const = uint_const(len(k["args"]))
        flags_const = uint_const(0)
        refl_block.extend(make_instr(OP_EXT_INST, [
            void_id, kernel_result, import_id, KERNEL,
            fn_id, name_id, num_args_const, flags_const, attrs_id,
        ]))

        # Args
        for a in k["args"]:
            ord_const = uint_const(a["ord"])
            arg_result = newid()
            if a["kind"] == "storage_buffer":
                set_const = uint_const(a["set"])
                bind_const = uint_const(a["binding"])
                refl_block.extend(make_instr(OP_EXT_INST, [
                    void_id, arg_result, import_id, ARG_STORAGE_BUFFER,
                    kernel_result, ord_const, set_const, bind_const,
                ]))
            elif a["kind"] == "pod_push_constant":
                offset_const = uint_const(a["offset"])
                size_const = uint_const(a["size"])
                refl_block.extend(make_instr(OP_EXT_INST, [
                    void_id, arg_result, import_id, ARG_POD_PUSH_CONSTANT,
                    kernel_result, ord_const, offset_const, size_const,
                ]))
            else:
                print(f"Unknown arg kind {a['kind']!r}", file=sys.stderr); sys.exit(3)

    # Phase G4 (TODO): emit per-device-global reflection. The natural opcode
    # is ProgramScopeVariablesStorageBuffer (28), which carries (set, binding,
    # init_data_str). clvk's reflection parser expects the third operand to
    # be a valid hex-byte string of size matching the descriptor; emitting a
    # symbol-name string instead makes clvk fail to JIT. The chipStar runtime
    # also currently does not consume this reflection. The bridging pass has
    # already encoded the symbol-name -> (set, binding) mapping in the IR
    # `!hipspv.device_globals` named-metadata; once Parts C+D land in the
    # runtime, they should either re-introduce this loop or thread the
    # mapping out-of-band (e.g. via a sidecar file emitted by the LLVM pass).
    #
    # For now, do nothing here so existing buffer-arg tests aren't regressed
    # by clvk reflection-parse failures. The downstream symptom is that
    # hipMemcpyToSymbol calls on these globals return hipErrorInvalidSymbol.
    pass

    # Find boundary: end of debug section (start of annotations).
    # Debug-section opcodes: OpSource* (3,4), OpName (5), OpMemberName (6),
    # OpString (7), OpModuleProcessed (330).
    # Annotations: OpDecorate (71), OpMemberDecorate (72), OpDecorationGroup (73)…
    debug_opcodes = {3, 4, 5, 6, 7, 330}
    # SPIR-V layout pre-debug: capability(17), extension(10), extinstimport(11),
    # memory model(14), entry point(15), execution mode(16), execution mode id(331).
    pre_debug_opcodes = {17, 10, 11, 14, 15, 16, 331}
    # Find the START of the debug section (so we insert OpStrings BEFORE
    # existing OpName instructions — spirv-val is strict about the order).
    debug_start_off = None
    annotations_off = None
    saw_debug = False
    for off, op, _wc, _ops in parse_instructions(words):
        if op in debug_opcodes and debug_start_off is None:
            debug_start_off = off
        if op in debug_opcodes:
            saw_debug = True
            continue
        if op in pre_debug_opcodes:
            continue
        if saw_debug:
            annotations_off = off
            break
    if debug_start_off is None:
        debug_start_off = annotations_off
    if annotations_off is None:
        # No debug section yet; fall back to inserting before first type/var.
        annotations_off = first_function_off if first_function_off is not None else end_off

    # Step 4: rebuild module. Layout per SPIR-V spec:
    #   1. capabilities
    #   2. extensions
    #   3. ExtInstImports        <- inject our import here
    #   4. memory model
    #   5. entry points
    #   6. execution modes
    #   7. debug
    #   8. annotations
    #   9. types/constants/global vars
    #  10. functions
    #
    # Reflection ExtInst calls live among the global section before functions
    # OR (clspv convention) at module end after functions. clvk reads at
    # parse-time, position is flexible. We'll insert after functions.

    # Find insertion points:
    # - "after type/constant section": before the first OpFunction.
    # - "after all functions": end of module.
    first_function_off = None
    end_off = len(words)
    for off, op, wc, _ops in parse_instructions(words):
        if op == 54:  # OpFunction
            first_function_off = off
            break
    # Insert ExtInstImport after the existing imports/extensions/capabilities.
    # Easy heuristic: insert before the first OpMemoryModel.
    # Per SPIR-V layout, OpExtension declarations must precede ALL
    # OpExtInstImport declarations, so we need a separate insertion point
    # for the OpExtension.
    OP_MEMORY_MODEL = 14
    OP_EXT_INST_IMPORT_OPCODE = 11
    memmodel_off = None
    first_extimport_off = None
    for off, op, wc, _ops in parse_instructions(words):
        if op == OP_EXT_INST_IMPORT_OPCODE and first_extimport_off is None:
            first_extimport_off = off
        if op == OP_MEMORY_MODEL:
            memmodel_off = off
            break
    if memmodel_off is None:
        print("No OpMemoryModel found — bailing.", file=sys.stderr)
        sys.exit(4)
    # Place OpExtension before the first existing OpExtInstImport (if any),
    # otherwise just before OpMemoryModel.
    extension_insert_off = (
        first_extimport_off if first_extimport_off is not None else memmodel_off
    )

    # Reassemble. SPIR-V layout (spec section 2.4):
    #   ExtInstImport goes before OpMemoryModel.
    #   OpString must live in the debug section (after exec modes,
    #     before annotations).
    #   Types/constants go after annotations.
    #   Reflection OpExtInst calls live among types/constants/globals
    #     (clvk parses them at any post-types point).
    out = []
    out.extend(words[:5])  # header
    # 5 .. extension_insert_off: capabilities + earlier extensions.
    out.extend(words[5:extension_insert_off])
    # OpExtension must precede ALL OpExtInstImport declarations.
    out.extend(ext_decl)
    # extension_insert_off .. memmodel: any existing OpExtInstImport(s).
    out.extend(words[extension_insert_off:memmodel_off])
    # OpExtInstImport for our reflection set, immediately before OpMemoryModel.
    out.extend(ext_import)
    # memmodel .. start-of-debug-section: entry points, exec modes
    out.extend(words[memmodel_off:debug_start_off])
    # OpStrings go FIRST in the debug section (spirv-val ordering quirk).
    out.extend(string_block)
    # rest of debug section + annotations up to first function
    boundary_types = first_function_off if first_function_off is not None else end_off
    out.extend(words[debug_start_off:boundary_types])
    # types / constants: inject our void/uint and uint constants here so the
    # later OpExtInst calls (post-function) have valid type/constant IDs.
    out.extend(inserts_after_types)
    out.extend(const_decls)
    # Functions
    out.extend(words[boundary_types:end_off])
    # Reflection OpExtInst calls live after function defs — that's where clspv
    # places them and where the function-ID forward references are valid.
    out.extend(refl_block)

    # Patch bound
    out[3] = bound + 1

    # Phase 4: rewrite WorkgroupSize OpVariable into spec-constant pattern.
    out, _ = rewrite_workgroup_size(out, out[3])

    write_spv(out_path, out)
    print(f"Wrote {out_path} (final bound: {out[3]})")

if __name__ == "__main__":
    main()
