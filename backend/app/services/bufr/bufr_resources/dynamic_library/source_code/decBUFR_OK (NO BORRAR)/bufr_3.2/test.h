/* A coordinate pair */

typedef struct point_s {
    varfl lat;      /* latitude */
    varfl lon;      /* longitude */
} point_t;


/* Meta information about image */

typedef struct meta_s {
    int year;
    int month;
    int day;
    int hour;
    int min;
    point_t radar;  /* Radar position */
    varfl radar_height;
} meta_t;

/* Level slicing table */

typedef struct scale_s {
    /* one method: */
    int nvals;       /* number of values in level slicing table */
    varfl vals[255]; /* scale values */

    /* another method: */
    varfl offset;    /* offset */
    varfl increment; /* increment */
} scale_t;

/* Radar image */

typedef struct img_s {
    int type;       /* Image type */
    varfl qual;     /* quality indicator */
    int grid;       /* Co-ordinate grid type */
    point_t nw;     /* Northwest corner of the image */
    point_t ne;     /* NE corner */
    point_t se;     /* SE corner */
    point_t sw;     /* SW corner */
    int nrows;      /* Number of pixels per row */
    int ncols;      /* Number of pixels per column */
    varfl psizex;   /* Pixel size along x coordinate */
    varfl psizey;   /* Pixel size along y coordinate */
    scale_t scale;  /* Level slicing table */
    unsigned short* data; /* Image data */
} img_t;

/* Projection information */

typedef struct proj_s {
    int type;       /* Projection type */
    varfl majax;    /* Semi-major axis or rotation ellipsoid */
    varfl minax;    /* Semi-minor axis or rotation ellipsoid */
    point_t orig;   /* Projection origin */
    int xoff;       /* False easting */
    int yoff;       /* False northing */
    varfl stdpar1;  /* 1st standard parallel */
    varfl stdpar2;  /* 2nd standard parallel */
} proj_t;


/* This is our internal data structure */

typedef struct radar_data_s {
    int wmoblock;           /* WMO block number */
    int wmostat;            /* WMO station number */
    meta_t meta;            /* Meta information about the product */
    img_t img;              /* Radar reflectivity image */
    proj_t proj;            /* Projection information */
    
} radar_data_t;

#ifndef BUFR_IO_H_INCLUDED
#define BUFR_IO_H_INCLUDED

int bufr_data_from_file(char* file, bufr_t* msg);
int bufr_data_to_file (char* file, char* imgfile, bufr_t* msg);
int bufr_z_decompress_to_mem (varfl **data, varfl* vals, int* nvals);
int bufr_z_compress_from_mem (varfl *data, int ndata, varfl* *vals, int* nvals);

#endif



#ifndef DESC_H_INCLUDED
#define DESC_H_INCLUDED

typedef double varfl;    /**< \brief Defines the internal float-variable type. 

                         Defines the internal float-variable type. 
                            This can
                            be float or double. Float needs less memory than
                            double. Double-floats need not to be converted by
                            your machine before operation (software runs 
                            faster). The default is double.
                            \note The format-string in all scanf-calls
                            must be changed for \p varfl-values !
                         */

/** This is the internal missing value indicator.
    Missing values are indicated as "missing" and if
    we find such a value we set it internally to
    MISSVAL
*/
#define MISSVAL 99999.999999

/*===========================================================================*/
/* definitions of data-structures                                            */
/*===========================================================================*/
/** \brief Holds the information contained in section 1

Holds the information contained in section 1
    \see bufr_sect_1_from_file, bufr_sect_1_to_file, bufr_encode_sections0125,
    bufr_decode_sections01
*/

typedef struct sect_1 {   
  int mtab;      /**<  \brief BUFR master table 

                 BUFR master table 
                 0 for standard WMO BUFR tables  */
  int subcent;   /**<  \brief Originating/generating subcenter */
  int gencent;   /**<  \brief Originating/generating center */
  int updsequ;   /**<  \brief Update sequence number 

                 Update sequence number 
                 zero for original BUFR
                    messages; incremented for updates */
  int opsec;     /**<  \brief optional section 

                 Bit 1 = 0 No optional section
                 = 1 Optional section included
                 Bits 2 - 8 set to zero (reserved) */
  int dcat;      /**<  \brief Data Category type (BUFR Table A) */
  int dcatst;    /**<  \brief Data Category sub-type 

                 Data Category sub-type 
                 defined by local ADP centres */
  int idcatst;   /**<  \brief International Data Category sub-type 

                 International Data Category sub-type 
                 Common Table C-13, used as of BUFR edition 4 */
  int vmtab;     /**<  \brief Version number of master tables used */
  int vltab;     /**<  \brief Version number of local tables used */
  int year;      /**<  \brief Year of century 

                 Year of century 
                 2 digit for BUFR edition < 4, 4 digit year as of BUFR edition
                 4  */
  int mon;       /**< \brief Month */
  int day;       /**< \brief Day */
  int hour;      /**< \brief Hour */
  int min;       /**< \brief Minute */
  int sec;       /**< \brief Second (used as of BUFR edition 4) */
} sect_1_t;

/** \brief Describes one data descriptor */

typedef struct dd {         
    int f; /**< \brief f*/
    int x; /**< \brief x*/
    int y; /**< \brief y*/
} dd;

/** \brief Defines an element descriptor */

typedef struct del {      
  dd d;                      /**< \brief Descriptor ID */
  char *unit;                /**< \brief Unit */
  int scale;                 /**< \brief Scale */
  varfl refval;              /**< \brief Reference Value */
  int dw;                    /**< \brief Data width (number of bits) */
  char *elname;              /**< \brief element name */
} del;

/** \brief Structure that defines a sequence of descriptors */

typedef struct dseq {        
  dd d;                      /**< \brief sequence-descriptor ID */
  int nel;                   /**< \brief Number of elements */
  dd *del;                   /**< \brief list of element descriptors */
} dseq;

/** \brief Structure that defines one descriptor. This can be an 
    element descriptor or a sequence descriptor */

typedef struct _desc {       
  int id;                    /**< \brief Can be \ref SEQDESC or \ref ELDESC */
  del *el;                   /**< \brief Element descriptor */
  dseq *seq;                 /**< \brief Sequence descriptor */
  int key;                   /**< \brief search key */
  int nr;                    /**< \brief serial number (insert position) */
} desc;


#define SEQDESC 0            /**< \brief Identifier for a sequence 
                                descriptor */
#define ELDESC  1            /**< \brief Identifier for an element 
                                descriptor */


/*===========================================================================*/
/* variables needed to hold data descriptors                                 */
/* If READDESC_MAIN is not defined all variables are declared as external.   */
/* So you sould define READDESC_MAIN only in one function. Otherwise you will*/
/* have this symbols multiple defined.                                       */
/*===========================================================================*/

#define MAXDESC   2000       /**< \brief Max. number of descriptors in the 
                                global descriptor-array (\ref des) */
#define OPTDESC   5          /* Number of optional descriptors at end */

#ifdef READDESC_MAIN
  int ndes;                 
  desc *des[MAXDESC+OPTDESC];
  int dw = 128;              
  int sc = 128;             
  int addfields = 0;        
  int ccitt_special = 0;     
  int cf_special = 0;       
  int add_f_special = 0;     
  int _desc_special = 0;     
#else

    /** \brief Total number of descriptors found */

    extern int ndes; 
    
    /** \brief Array holding all data descriptors 
                                        
    Array holding all data descriptors. 
    The descriptors are read from the descriptor table files 
    using \ref read_tables or \ref read_tab_b and read_tab_d

    \see read_tables, read_tab_b, read_tab_d, get_index
    */
    extern desc *des[MAXDESC+OPTDESC];

    /** \brief Current data width modification factor (default: 128)

    Current data width modification factor (default: 128)
    Add dw - 128 to the data-width   
    (dw can be optionally set by 2 01 YYY) 
    */
    extern int dw;

    /** \brief Current scale modification factor (default: 128)

    Current scale modification factor (default: 128).
    Add sc - 128 to the scale-factor
    (sc can be optionally set by 2 02 YYY) 
    */
    extern int sc;

    /** \brief Number of associated fields to be added to 
        any data-item. 
        
        Number of associated fields to be added to any data-item.
        \p addfields can be set by 2 04 YYY and canceled by 
        2 04 000 

    */
    extern int addfields;

    /** \brief Special index for ccitt characters.
        
    This index is used by \ref bufr_parse_new and its derivates
    to indicate that a value is a CCITT character

    \see bufr_parse_new, cbin, cbout
    */
    extern int ccitt_special;

    /* \brief Special index for change of reference field.
       
    \todo implement this
    */
    extern int cf_special;

    /** \brief Special index for associated fields.

    This index is used by \ref bufr_parse_new and its derivates
    to indicate that a value is an associated field.
    
    \see bufr_parse_new, cbin, cbout
    */
    extern int add_f_special;

    /** \brief Special index for descriptors without data.
    
    This index is used by \ref bufr_parse_new and its derivates
    to indicate that we have a descriptor without value for output.
    
    \see bufr_parse_new, cbout
    */
    extern int _desc_special;

#endif

/*===========================================================================*/
/* The following definition will be used to have either                      */
/* function-prototyping in ANSI-C e.g.: void abc (int a, int b);   or        */
/* Kernighan-Ritchie-prototyping link   void abc ();                         */
/*===========================================================================*/

#if defined (NON_ANSI)
#define P0
#define P1(a)                
#define P2(a,b)              
#define P3(a,b,c)            
#define P4(a,b,c,d)          
#define P5(a,b,c,d,e)        
#define P6(a,b,c,d,e,f)      
#define P7(a,b,c,d,e,f,g)    
#define P8(a,b,c,d,e,f,g,h)  
#else
#define P0                   void
#define P1(a)                a
#define P2(a,b)              a,b
#define P3(a,b,c)            a,b,c
#define P4(a,b,c,d)          a,b,c,d
#define P5(a,b,c,d,e)        a,b,c,d,e
#define P6(a,b,c,d,e,f)      a,b,c,d,e,f
#define P7(a,b,c,d,e,f,g)    a,b,c,d,e,f,g
#define P8(a,b,c,d,e,f,g,h)  a,b,c,d,e,f,g,h
#endif

/* for compilers having SEEK_CUR and SEEK_SET not defined: */

#ifndef SEEK_SET
#define SEEK_SET 0
#endif
#ifndef SEEK_END
#define SEEK_END 2
#endif

/*===========================================================================*/
/* function prototype                                                        */
/*===========================================================================*/

int read_tab_b (char *fname);
int read_tab_d (char *fname);
char *get_unit (dd *d);
int get_index (int typ, dd *d);
void free_descs (void);
void trim (char *buf);
int read_tables (char *dir, int vm, int vl, int subcenter, int gencenter);
void show_desc (int f, int x, int y);
void show_desc_args (int argc, char **argv);
int desc_is_codetable (int ind);
int desc_is_flagtable (int ind);
int read_bitmap_tables (char *dir,  int vltab, int gencent, int subcent);
int check_bitmap_desc (dd *d);

#endif

/* end of file */


/** \file bufr.h
    \brief Definitions of main OPERA BUFR library functions
    
    This file contains declaration of functions used for encoding and 
    decoding data to BUFR format.
*/

/*===========================================================================*/
/* global variables                                                          */
/* If BUFR_MAIN is not defined all variables are declared as external.       */
/* So you sould define BUFR_MAIN only in one function. Otherwise you will    */
/* have this symbols multiple defined.                                       */
/*===========================================================================*/

#ifdef BUFR_MAIN

int _bufr_edition = 4;      /**< bufr edition number */
int _opera_mode = 0;        /* input and output bitmaps to / from file */
int _replicating = 0;       /**< indicates a data replication process */
int _subsets = 1;           /**< number of subsets */

#else
/** \brief global bufr edition number 

   The bufr edition number is stored in section 0 of the BUFR message.
   It is used by the software for determining the format of section 1.

   \see bufr_get_date_time, bufr_encode_sections0125, bufr_decode_sections01,
        bufr_parse_new, bufr_val_from_datasect, bufr_val_to_datasect
*/
extern int _bufr_edition;   
extern int _opera_mode;

/** \brief global replication indicator

    This flag is used to indicate an ongoing data replication and is set
    by \ref bufr_parse_new. It can be used for different output formating 
    when a replication occurs.

    \see bufr_parse_new, bufr_file_out

*/

extern int _replicating;

extern int _subsets;
#endif

#ifndef BUFR_H_INCLUDED
#define BUFR_H_INCLUDED

#define MAX_DESCS 1000       /**< \brief Maximum number of data descriptors
                                in a BUFR message */

#define MEMBLOCK   100       /* The memory-area holding data-strings 
                                and data-descriptors is allocated and 
                                reallocated in blocks of MEMBLOCK elements. */

/*===========================================================================*/
/* structures                                                                */
/*===========================================================================*/

/** \brief Structure that holds the encoded bufr message */
typedef struct bufr_s {      

    char* sec[6];            /**< \brief pointers to sections */
    int   secl[6];           /**< \brief length of sections */
} bufr_t;

typedef char* bd_t;          /**< \brief one bufr data element is a string */

/** \brief Structure holding values for callbacks 
    \ref bufr_val_from_global and \ref bufr_val_to_global */

typedef struct bufrval_s { 
    varfl* vals;            /**< \brief array of values */
    int vali;        /**< \brief current index into array of values */
    int nvals;       /**< \brief number of values */
} bufrval_t;



/*===========================================================================*/
/* protypes of functions in BUFR.C                                           */
/*===========================================================================*/

/* basic functions for encoding to BUFR */

int bufr_create_msg (dd *descs, int ndescs, varfl *vals, void **datasec, 
                        void **ddsec, size_t *datasecl, size_t *ddescl);
int bufr_encode_sections34 (dd* descs, int ndescs, varfl* vals, bufr_t* msg);
int bufr_encode_sections0125 (sect_1_t* s1, bufr_t* msg);
int bufr_write_file (bufr_t* msg, const char* file);

/* basic function for decoding from BUFR */

int bufr_read_file (bufr_t* msg, const char* file);
int bufr_get_sections (char* bm, int len, bufr_t* msg);
int bufr_decode_sections01 (sect_1_t* s1, bufr_t* msg);
int bufr_read_msg (void *datasec, void *ddsec, size_t datasecl, size_t ddescl,
                      dd **desc, int *ndescs, varfl **vals, size_t *nvals);

/* extended functions for encoding to BUFR */

void bufr_sect_1_from_file (sect_1_t* s1, const char* file);
int bufr_open_descsec_w (int subsets);
int bufr_out_descsec (dd *descp, int ndescs, int desch);
void bufr_close_descsec_w(bufr_t* bufr, int desch);
int bufr_parse_in  (dd *descs, int start, int end, 
                    int (*inputfkt) (varfl *val, int ind),
                    int callback_descs);

/* extended functions for decoding from BUFR */

int bufr_open_descsec_r (bufr_t* msg, int *subsets);
int bufr_get_ndescs (bufr_t* msg);
int bufr_in_descsec (dd** descs, int ndescs, int desch);
void bufr_close_descsec_r(int desch);
int bufr_parse_out  (dd *descs, int start, int end, 
                     int (*outputfkt) (varfl val, int ind),
                     int callback_all_descs);
int bufr_sect_1_to_file (sect_1_t* s1, const char* file);

/* utility functions */

void bufr_free_data (bufr_t* d);
int bufr_check_fxy(dd *d, int ff, int xx, int yy);
void bufr_get_date_time (long *year, long *mon, long *day, long *hour,
                            long *min);
int bufr_val_to_array (varfl **vals, varfl v, int *nvals);
int bufr_desc_to_array (dd* descs, dd d, int* ndescs);
int bufr_parse_new (dd *descs, int start, int end, 
                    int (*inputfkt) (varfl *val, int ind),
                    int (*outputfkt) (varfl val, int ind),
                    int callback_all_descs);
int bufr_parse (dd *descs, int start, int end, varfl *vals, unsigned *vali,
                int (*userfkt) (varfl val, int ind));
bufrval_t* bufr_open_val_array ();
void bufr_close_val_array ();
int bufr_open_datasect_w ();
void bufr_close_datasect_w(bufr_t* msg);
int bufr_open_datasect_r (bufr_t* msg);
void bufr_close_datasect_r();

/* callback functions for use with bufr_parse_* */

int bufr_val_from_global (varfl *val, int ind);
int bufr_val_to_global (varfl val, int ind);

/* deprecated functions */

void bufr_clean ();
int val_to_array (varfl **vals, varfl v, size_t *nvals);
int setup_sec0125 (char *sec[], size_t secl[], sect_1_t s1);
int save_sections (char *sec[], size_t secl[], char *buffile);

#endif



/*===========================================================================*/
/* function prototypes                                                       */
/*===========================================================================*/

#ifndef BITIO_H_INCLUDED
#define BITIO_H_INCLUDED

int bitio_i_open (void *buf, size_t size);
int bitio_i_input (int handle, unsigned long *val, int nbits);
size_t bitio_o_get_size (int handle);
void bitio_i_close (int handle);
int bitio_o_open ();
long bitio_o_append (int handle, unsigned long val, int nbits);
void bitio_o_outp (int handle, unsigned long val, int nbits, long bitpos);
void *bitio_o_close (int handle, size_t *nbytes);

#endif

/* end of file */



#ifndef RLENC_H_INCLUDED
#define RLENC_H_INCLUDED

int rlenc_from_file (char* infile, int nrows, int ncols, varfl** vals, 
                     int* nvals, int depth);
int rlenc_from_mem (unsigned short* img, int nrows, int ncols, varfl** vals, 
                    int* nvals);
int rldec_to_file (char* outfile, varfl* vals, int depth, int* nvals);
int rldec_to_mem (varfl* vals, unsigned short* *img, int* nvals, int* nrows,
                  int* ncols);
int rlenc_compress_line_new (int line, unsigned int* src, int ncols, 
                             varfl** dvals, int* nvals);
void rldec_decompress_line (varfl* vals, unsigned int* dest, int* ncols, 
                            int* nvals);
void rldec_get_size (varfl* vals, int* nrows, int* ncols);

/* float methods */
int rlenc_from_mem_float (float* img, int nrows, int ncols, varfl** vals, 
                          int* nvals);
int rldec_to_mem_float (varfl* vals, float* *img, int* nvals, int* nrows,
                        int* ncols);
int rlenc_compress_line_float (int line, float* src, int ncols, 
                               varfl** dvals, int* nvals);
void rldec_decompress_line_float (varfl* vals, float* dest, int* ncols, 
                                  int* nvals);

/* old functions */
int rlenc (char *infile, int nrows, int ncols, varfl **vals, size_t *nvals);
int rldec (char *outfile, varfl *vals, size_t *nvals);
int rlenc_compress_line (int line, unsigned char *src, int ncols, 
                         varfl **dvals, size_t *nvals);



#endif

/* end of file */
