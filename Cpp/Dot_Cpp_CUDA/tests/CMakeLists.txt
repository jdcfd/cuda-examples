add_executable(test_dot
    test_dot.cu
)

target_link_libraries(test_dot PRIVATE 
    dot_lib 
    Catch2::Catch2WithMain
)

# This is necessary for Catch2 to discover tests automatically
include(Catch)
catch_discover_tests(test_dot)
