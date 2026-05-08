install(FILES inject_reflection.py
        DESTINATION share/hipspv
        COMPONENT hipspv-inject-reflection
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE)

if(NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(install-hipspv-inject-reflection
                           COMPONENT hipspv-inject-reflection)
endif()
