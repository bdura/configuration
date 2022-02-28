# Overview

This project aims at providing an abstraction over Thinc's powerful configuration system, as well as catalogue's registry system.

## Rationale

The people at explosion have made tremendous efforts to provide a usable configuration and registry system. However, I found there were a few points missing:

1. The code that connects the dots is hidden deep into SpaCy and Thinc, making boilerplate code necessary at the start of each new project
2. The registry system lacks a unified resolution system
