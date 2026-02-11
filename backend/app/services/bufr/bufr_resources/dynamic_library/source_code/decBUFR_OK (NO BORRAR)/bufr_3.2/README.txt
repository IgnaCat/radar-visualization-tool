-------------------------------------------------------------------------------
OPERA FM94-BUFR Encoding and Decoding Software Version 3.2
Binary release for Linux
-------------------------------------------------------------------------------

Included Files:
---------------

encbufr                 bufr encoding software
decbufr                 bufr decoding software
bufrtab*.csv            WMO BUFR tables
localtab*.csv           OPERA BUFR tables
bmtab*.csv              OPERA bitmap tables

Usage:
------

- for encoding:

 encbufr [-d tabdir] input_file output_file

- for decoding:

 decbufr [-d tabdir] input_file output_file [image_file]

where tabdir is the location of the BUFR tables.

Documentation:
--------------

For detailed documentation please refer to the OPERA BUFR software
documentation (bufr_sw_desc.pdf) which is available as separate download
and is also included in the full (source code) release of this sofware.

Release notes:
--------------

Please note that the BUFR tables are a result of other ongoing OPERA 
activities. The tables included in this distribution are the most recent 
available as of December 2012. In the future new tables may be released 
by OPERA, so be sure to check the OPERA home page for updated BUFR tables
before using them operationally.

With the included section.1 file, local tables 8 and originating centre 247
are used for encoding (this is also the default if no section.1 file
is present). Please change this file according to your needs.

The new software version uses BUFR Edition 4 as default for encoding, 
unless overridden by the command line option -e3. BUFR Edition 4 should 
present no problems to any BUFR decoder. The edition 4 is already 
supported (for decoding) by the BUFR software version 3.0, which is 
available since 5 years.


What's new in this version: 


- implementation of BUFR Edition 4 modification descriptors:
  2 7 y (change of scale, width and refence value)
  2 8 y (change width of ascii field)
  2 41 y (event)
  2 42 y (conditioning event)
  2 43 y (categorical forecast)

- added two functions to z-compress data values from memory or to 
  z-decompress to memory (see bufr_io.h and bufr_io.c)

- use BUFR edition 4 as default for encoding (can be changed per 
  command line option)

- use WMO tables version 14 and OPERA tables 247 - version 8 as default 
  for encoding (can be changed via section.1 file)

- encbufr option -e3 to encode with BUFR edition 3

- documentation update (BUFR software description, API documentation)

--
