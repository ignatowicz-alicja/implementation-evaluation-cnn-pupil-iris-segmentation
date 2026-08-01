# Third-party iris-recognition implementation

This directory is reserved for the external implementation called by the project evaluation wrappers.

Expected layout:

```text
third_party/Iris-Recognition-master/
├── python/
│   └── fnc/
│       ├── boundary.py
│       ├── segment.py
│       ├── normalize.py
│       ├── encode.py
│       └── matching.py
└── matlab/
    └── fnc/
        └── ... original MATLAB functions ...
```

## Ownership

Files placed in this directory are **not authored by the maintainers of this repository**. They belong to their original authors or port maintainers and retain their original copyright, notices, citation requirements, and usage restrictions.

The classical method is based on:

> Libor Masek and Peter Kovesi, *MATLAB Source Code for a Biometric Identification System Based on Iris Patterns*, The University of Western Australia, 2003.

Official project page: <https://www.peterkovesi.com/studentprojects/libor/>

Before publishing any third-party files on GitHub, verify that their license or terms permit redistribution. Preserve all original headers and notices. The safest public setup is to leave this directory empty and document how users can obtain the dependency independently.
