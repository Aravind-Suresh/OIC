# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/cmake-gui

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/aravind/Downloads/dlib-18.16/examples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/aravind/Downloads/dlib-18.16/examples/build

# Include any dependencies generated for this target.
include CMakeFiles/quantum_computing_ex.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/quantum_computing_ex.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/quantum_computing_ex.dir/flags.make

CMakeFiles/quantum_computing_ex.dir/quantum_computing_ex.cpp.o: CMakeFiles/quantum_computing_ex.dir/flags.make
CMakeFiles/quantum_computing_ex.dir/quantum_computing_ex.cpp.o: ../quantum_computing_ex.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/aravind/Downloads/dlib-18.16/examples/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/quantum_computing_ex.dir/quantum_computing_ex.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/quantum_computing_ex.dir/quantum_computing_ex.cpp.o -c /home/aravind/Downloads/dlib-18.16/examples/quantum_computing_ex.cpp

CMakeFiles/quantum_computing_ex.dir/quantum_computing_ex.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quantum_computing_ex.dir/quantum_computing_ex.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/aravind/Downloads/dlib-18.16/examples/quantum_computing_ex.cpp > CMakeFiles/quantum_computing_ex.dir/quantum_computing_ex.cpp.i

CMakeFiles/quantum_computing_ex.dir/quantum_computing_ex.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quantum_computing_ex.dir/quantum_computing_ex.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/aravind/Downloads/dlib-18.16/examples/quantum_computing_ex.cpp -o CMakeFiles/quantum_computing_ex.dir/quantum_computing_ex.cpp.s

CMakeFiles/quantum_computing_ex.dir/quantum_computing_ex.cpp.o.requires:
.PHONY : CMakeFiles/quantum_computing_ex.dir/quantum_computing_ex.cpp.o.requires

CMakeFiles/quantum_computing_ex.dir/quantum_computing_ex.cpp.o.provides: CMakeFiles/quantum_computing_ex.dir/quantum_computing_ex.cpp.o.requires
	$(MAKE) -f CMakeFiles/quantum_computing_ex.dir/build.make CMakeFiles/quantum_computing_ex.dir/quantum_computing_ex.cpp.o.provides.build
.PHONY : CMakeFiles/quantum_computing_ex.dir/quantum_computing_ex.cpp.o.provides

CMakeFiles/quantum_computing_ex.dir/quantum_computing_ex.cpp.o.provides.build: CMakeFiles/quantum_computing_ex.dir/quantum_computing_ex.cpp.o

# Object files for target quantum_computing_ex
quantum_computing_ex_OBJECTS = \
"CMakeFiles/quantum_computing_ex.dir/quantum_computing_ex.cpp.o"

# External object files for target quantum_computing_ex
quantum_computing_ex_EXTERNAL_OBJECTS =

quantum_computing_ex: CMakeFiles/quantum_computing_ex.dir/quantum_computing_ex.cpp.o
quantum_computing_ex: CMakeFiles/quantum_computing_ex.dir/build.make
quantum_computing_ex: dlib_build/libdlib.a
quantum_computing_ex: /usr/lib/x86_64-linux-gnu/libpthread.so
quantum_computing_ex: /usr/lib/x86_64-linux-gnu/libnsl.so
quantum_computing_ex: /usr/lib/x86_64-linux-gnu/libSM.so
quantum_computing_ex: /usr/lib/x86_64-linux-gnu/libICE.so
quantum_computing_ex: /usr/lib/x86_64-linux-gnu/libX11.so
quantum_computing_ex: /usr/lib/x86_64-linux-gnu/libXext.so
quantum_computing_ex: /usr/lib/x86_64-linux-gnu/libpng.so
quantum_computing_ex: /usr/lib/x86_64-linux-gnu/libjpeg.so
quantum_computing_ex: /usr/lib/libblas.so
quantum_computing_ex: /usr/lib/liblapack.so
quantum_computing_ex: CMakeFiles/quantum_computing_ex.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable quantum_computing_ex"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/quantum_computing_ex.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/quantum_computing_ex.dir/build: quantum_computing_ex
.PHONY : CMakeFiles/quantum_computing_ex.dir/build

CMakeFiles/quantum_computing_ex.dir/requires: CMakeFiles/quantum_computing_ex.dir/quantum_computing_ex.cpp.o.requires
.PHONY : CMakeFiles/quantum_computing_ex.dir/requires

CMakeFiles/quantum_computing_ex.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/quantum_computing_ex.dir/cmake_clean.cmake
.PHONY : CMakeFiles/quantum_computing_ex.dir/clean

CMakeFiles/quantum_computing_ex.dir/depend:
	cd /home/aravind/Downloads/dlib-18.16/examples/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/aravind/Downloads/dlib-18.16/examples /home/aravind/Downloads/dlib-18.16/examples /home/aravind/Downloads/dlib-18.16/examples/build /home/aravind/Downloads/dlib-18.16/examples/build /home/aravind/Downloads/dlib-18.16/examples/build/CMakeFiles/quantum_computing_ex.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/quantum_computing_ex.dir/depend
