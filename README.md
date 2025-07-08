# Machine learning project 001

## Project setup :

First of all, thanks to **@kigster** for this beautiful template : 
[template link](https://github.com/kigster/cmake-project-template)

### Installing dependency 

- Download a version of libtorch like said here : https://pytorch.org/get-started/locally/
- Link to libtorch to the project : 
  - add the argument : `-DCMAKE_PREFIX_PATH="<path to libtorch/share/cmake>"` to your cmake command
- use vcpkg to install : `apache arrow` with parquet and openCV 
- add `-DCMAKE_TOOLCHAIN_FILE="<Path to vcpkg>/vcpkg/scripts/buildsystems/vcpkg.cmake"` to your config profile
