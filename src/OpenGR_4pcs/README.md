# OpenGR

### fork from https://github.com/STORM-IRIT/OpenGR

OpenGR is a set C++ libraries for 3D Global Registration.
It is a fork of the [Super4PCS library](https://github.com/nmellado/Super4PCS), and aims at providing several state of the art global registration algorithms for 3d data.
This fork is maintained by the same authors as the Super4PCS library.

See the offical documentation here: https://storm-irit.github.io/OpenGR/index.html

********
Here's a BibTeX entry for OpenGR that you can use in your academic publications:

```
 @MISC{openGR,
  author = {Nicolas Mellado and others},
  title = {OpenGR: A C++ library for 3D Global Registration},
  howpublished = {https://storm-irit.github.io/OpenGR/},
  year = {2017}
 }
 ```

********

Compile use CMake.


***To run example:***
```
cd build/

./demos/Super4PCS/Super4PCS  -i [OBJFILE_SRC] [OBJFILE_DST] -o 0.2 -d 0.01 -n 500

```

Or run using PCL wrapper, file can be OBJ or PLY:
```
./demos/PCLWrapper/OpenGR-PCLWrapper [FILE_SRC] [FILE_DST] -o 0.2 -d 0.01 -n 500
```
