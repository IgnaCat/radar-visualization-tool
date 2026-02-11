#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <errno.h>
#include <ctype.h>
#include <time.h>
#include <memory.h>


/* rlenc */

#define LBUFLEN 5000         /**< \brief Size of the internal buffer holding 
                                one uncompressed line */
#define ENCBUFL 5000         /**< \brief Size of the internal buffer holding 
                                one compressed line */

/*===========================================================================*/
/** \ingroup deprecated_g

    \deprecated Use \ref rlenc_from_file instead.
    \brief Runlength-encodes a radar image


    This function encodes a "one byte per pixel" radar image to BUFR runlength-
   code and stores the resulting values by a call to VAL_TO_ARRAY.

   \param[in] infile   File holding the "one byte per pixel" radar image.
   \param[in] ncols    Number of columns of the image.
   \param[in] nrows    Number of rows of the image.
   \param[in,out] vals     Float-array holding the coded image.
   \param[in,out] nvals    Number of values in VALS.

   \return The return-value ist 1 on success, 0 on a fault.
*/

int rlenc (char* infile, int nrows, int ncols, varfl** vals, size_t* nvals)

{
  FILE *fp;
  unsigned char buf[LBUFLEN];
  int i, n;

/* check if the internal buffer is large enough to hold one uncompressed
         line */

  assert (ncols <= LBUFLEN);

/* open file holding the radar image */

  fp = fopen (infile, "rb");
  if (fp == NULL) {
    fprintf (stderr, "error opening '%s'\n", infile);
    return 0;
  }

/* output number of rows */

  val_to_array (vals, (varfl) nrows, nvals);  

/* compress line by line */

  for (i = 0; i < nrows; i ++) {
    n = fread (buf, 1, ncols, fp);
    if (n != ncols) {
      fprintf (stderr, "read error from file '%s'\n", infile);
      goto err;
    }
    if (!rlenc_compress_line (i, buf, ncols, vals, nvals)) goto err;
  }
  fclose (fp);
  return 1;

err:
  fclose (fp);
  return 0;
}


/*===========================================================================*/
/** \ingroup deprecated_g
    \deprecated Use \ref rlenc_compress_line_new instead.

    \brief Encodes one line of a radar image to BUFR runlength-code

    This function encodes one line of a radar image to BUFR runlength-code and
    stores the resulting values by a call to \ref val_to_array.

    \param[in] line     Line number.
    \param[in] src      Is where the uncompressed line is stored.
    \param[in] ncols    Number of pixels per line.
    \param[in,out] dvals     Float-array holding the coded image.
    \param[in,out] nvals    Number of values in VALS.
   
    \return The function returns 1 on success, 0 on a fault.
*/

int rlenc_compress_line (int line, unsigned char* src, int ncols, 
                         varfl** dvals, size_t* nvals)


{
  int count, i, n, npar, lens[LBUFLEN], cw, ncgi, nngi = 0;
  unsigned char val, lval = 0, vals[LBUFLEN];
  varfl encbuf[ENCBUFL];

  /* compress Line into a runlength format */

  count = n = 0;
  for (i = 0; i < ncols; i ++) {
    val = *(src + i);
    if (i != 0 && (val != lval || count >= 255)) {  /* (n >= 255) to ensure that BUFR-descriptor 0 31 001 does not exceed */
      lens[n] = count;
      vals[n] = lval;
      n ++;
      count = 0;
      lval = val;
    }
    lval = val;
    count ++;
  }
  lens[n] = count;
  vals[n] = lval;
  n ++;
  

  /* line is runlength-compressed now to N parts, each of them identified 
     by a length (LENS) and a value (VALS). */
    

  /* Count number of parcels. One parcel is identified by a COUNT of 1
     followed by a COUNT > 1 */

  npar = 0;
  for (i = 0; i < n - 1; i ++) if (lens[i] == 1 && lens[i+1] > 1) npar ++;
  npar ++;

  /* output line-number */

  for (i = 0; i < ENCBUFL; i ++) encbuf[i] = (varfl) 0.0;
  cw = 0;
  encbuf[cw++] = line;  

  /* compress it to parcels */

  encbuf[cw++] = npar;                   /* number of parcels */

  ncgi = cw ++;                          /* is where the number of compressable groups is stored */
  encbuf[ncgi] = (varfl) 0.0;            /* number of compressable groups */

  i = 0;
  for (i = 0; i < n; i ++) {
    if (lens[i] > 1) {                   /* compressable group found */
      if (i > 0 && lens[i-1] == 1) {     /* A new parcel starts here */
        ncgi = cw ++;                    /* is where the number of compressable groups is stored */
        encbuf[ncgi] = (varfl) 0.0;      /* number of compressable groups */
      }
      encbuf[ncgi] += 1.0;
      encbuf[cw++] = lens[i];
      encbuf[cw++] = vals[i];
    }
    else {                               /* non compressable group found */
      if (i == 0 || lens[i-1] != 1) {    /* this is the first uncompressable group in the current parcel */
        nngi = cw ++;                    /* is where the number of non compressable groups is stored */
        encbuf[nngi] = (varfl) 0.0;      /* Number of non compressable groups */
      }
      encbuf[nngi] += 1.0;
      encbuf[cw++] = vals[i];
    }
  }
  if (lens[n-1] != 1) encbuf[cw++] = 0;  /* number of noncompressable groups in the last parcel = 0 */
  assert (cw <= ENCBUFL);

  /* compresson to parcels finished */


  for (i = 0; i < cw; i ++)
      if (!val_to_array (dvals, encbuf[i], nvals)) return 0;

  /* Output data for debugging purposes */

  /*cw = 0;
  printf ("\n\nline no. %d:\n", (int) encbuf[cw++]);
  npar = (int) encbuf[cw];
  printf ("number of parcels: %d\n", (int) encbuf[cw++]);
  for (i = 0; i < npar; i ++) {
    ncgi = (int) encbuf[cw];
    printf ("number of compressable groups: %d\n", (int) encbuf[cw++]);
    for (j = 0; j < ncgi; j ++) {
      printf ("count: %d\n", (int) encbuf[cw++]);
      printf ("val: %d\n", (int) encbuf[cw++]);
    }
    nngi = (int) encbuf[cw];
    printf ("number of uncompressable pixels: %d\n", (int) encbuf[cw++]);
    for (j = 0; j < nngi; j ++) {
      printf ("val: %d\n", (int) encbuf[cw++]);
    }
  }*/

  return 1;
}

/*===========================================================================*/
/** \ingroup deprecated_g
    \deprecated Use \ref rldec_to_file instead.

    \brief Decodes a BUFR-runlength-encoded radar image

    This function decodes a BUFR-runlength-encoded radar image stored at
    \p VALS. The decoded image is stored in a one "byte-per-pixel-format" at
    the file \p OUTFILE.

    \param[in] outfile  Destination-file for the "one byte per pixel" radar 
                        image.
    \param[in] vals     Float-array holding the coded image.
    \param[in] nvals    Number of values needed for the radar image.

    \return The return-value ist 1 on success, 0 on a fault.
*/

int rldec (char* outfile, varfl* vals, size_t* nvals)

{
  FILE *fp;
  int i, j, k, l, ngr, nrows, npar, val, count, nup;
  varfl *ovals;

/* Open destination-file for output */

  fp = fopen (outfile, "wb");
  if (fp == NULL) return 0;

/* decode line by line */

  ovals = vals;
  nrows = (int) *vals ++;               /* number of rows */
  for (i = 0; i < nrows; i ++) {        /* loop for lines */
      vals ++;                            /* skip linenumber */
      npar = (int) *vals ++;              /* number of parcels */
      for (j = 0; j < npar; j ++) {       /* loop for parcels */
          ngr = (int) *vals ++;             /* number of compressable groups */
          for (k = 0; k < ngr; k ++) {      /* loop for compressable groups */
              count = (int) *vals ++;
              val =   (int) *vals ++;
              for (l = 0; l < count; l ++) {  /* loop for length of group */
                  fputc (val, fp);
              }
          }
          nup = (int) *vals ++;             /* number of uncompressable pixels */
          for (k = 0; k < nup; k ++) {      /* loop for uncompressable pixels */
              val = (int) *vals ++;
              fputc (val, fp);
          }
      }
  }

  /* close file */

  fclose (fp);

  /* calculate number of values in VALS occupied by the radar image */

  *nvals = vals - ovals;
  return 1;
}

/*===========================================================================*/
/* New functions */
/*===========================================================================*/

/** \ingroup rlenc_g

    \brief Runlength-encodes a radar image from a file to an array

    This function encodes a radar image file with \p depth bytes per pixel
    to BUFR runlength-code and stores the resulting values into an
    array \p vals by a call to \ref bufr_val_to_array. 

    Currently \p depth can be one or two bytes per pixel.
    In case of two bytes per pixel data is read in 
    "High byte - low byte order". So pixel values 256 257 32000 are represented
    by 0100 0101 7D00 hex.  

    \note In difference to the old \ref rlenc function the
    initial length of \p vals must be given in the parameter
    \p nvals in order to prevent \ref bufr_val_to_array from writing to an
    arbitrary position.

    \param[in] infile   File holding the radar image.
    \param[in] ncols    Number of columns of the image.
    \param[in] nrows    Number of rows of the image.
    \param[in] depth    Image depth in bytes
    \param[in,out] vals     Float-array holding the coded image.
    \param[in,out] nvals    Number of values in VALS.

    \return The return-value ist 1 on success, 0 on a fault.

    \see rlenc_from_mem, rldec_to_file, rlenc_compress_line_new
*/
int rlenc_from_file (char* infile, int nrows, int ncols, varfl* *vals, 
                     int *nvals, int depth)

{
    FILE *fp;
    unsigned char cbuf[LBUFLEN * 2];
    unsigned int ibuf[LBUFLEN];
    float fbuf[LBUFLEN];
    int i, n, j, ok;
/*
    if (depth > 4) {
        fprintf (stderr, 
                 "Unsupported number of bits per bixel!\n");
        return 0;
    }
*/
    /* check if the internal buffer is large enough to hold one uncompressed
       line */

    if (ncols > LBUFLEN) {
        fprintf (stderr, "ERROR: Number of columns larger than %d!\n",
                 LBUFLEN);
        return 0;
    }

    /* open file holding the radar image */

    fp = fopen (infile, "rb");
    if (fp == NULL) {
        fprintf (stderr, "error opening '%s'\n", infile);
        return 0;
    }

    /* read values as float from file */ 

    if (depth > 4) 
    {
        while ((n = fread (fbuf, sizeof(float), LBUFLEN, fp)) > 0)
        {
            for (i = 0; i< n; i++)
                bufr_val_to_array (vals, fbuf[i], nvals);
        }
        fclose(fp);
        return 1;
    }

    /* read P5 header for pgm-format */

    if (strstr (infile, ".pgm") != NULL || strstr (infile, ".PGM"))
    {
        fscanf (fp, "P5 %d %d %d ", &j, &i, &n);
        if (i != nrows || j != ncols || (n > 255 && depth < 2)) 
        {
            fprintf (stderr, "error in pgm file '%s'\n", infile);
            return 0;
        }
    }

    /* output number of rows */

    bufr_val_to_array (vals, (varfl) nrows, nvals);  

    /* compress line by line */

    for (i = 0; i < nrows; i ++) {

        /* read row from file */

        if (depth == 4)
            n = fread (fbuf, 1, ncols * depth, fp);
        else
            n = fread (cbuf, 1, ncols * depth, fp);
        if (n != ncols * depth) {
            fprintf (stderr, "read error from file '%s'\n", infile);
            fclose (fp);
            return 0;
        }
            
        /* convert to integer */
        
        if (depth == 1) {
            for (j = 0; j < ncols; j ++)
                ibuf[j] = (unsigned int) cbuf[j];
        } else if (depth == 2) {
            for (j = 0; j < ncols; j++)
                ibuf[j] = (cbuf[j*2] << 8) + cbuf[j*2+1];
        }

        /* compress line to varfl array */

        if (depth == 4)
            ok =rlenc_compress_line_float (i, fbuf, ncols, vals, nvals);
        else
            ok =rlenc_compress_line_new (i, ibuf, ncols, vals, nvals);
    }
    fclose (fp);
    return ok;
}

/*===========================================================================*/
/** \ingroup rldec_g

    \brief Decodes a BUFR-runlength-encoded radar image to a file

    This function decodes a BUFR-runlength-encoded radar image stored at
    \p vals. The decoded image is stored in a "\p depth byte-per-pixel-format"
    at the file \p outfile.
    Currently \p depth can be one or two bytes per pixel.
    In case of two bytes per pixel data is stored in 
    "High byte - low byte order". So pixel values 256 257 32000 are represented
    by 0100 0101 7D00 hex.  

   \param[in] outfile  Destination-file for the radar image.
   \param[in] vals     Float-array holding the coded image.
   \param[in] depth    Number of bytes per pixel
   \param[out] nvals   Number of \ref varfl values needed for the 
                       compressed radar image.

   \return The return-value ist 1 on success, 0 on a fault.

   \see rldec_to_mem, rldec_decompress_line, rlenc_from_file
*/

int rldec_to_file (char* outfile, varfl* vals, int depth, int* nvals)

{
    FILE *fp;
    int i, j, nrows, ncols, nc, nv;
    unsigned int ibuf[LBUFLEN];
    unsigned char cbuf[LBUFLEN*2];
    float fbuf[LBUFLEN];
    varfl *ovals;
/*
    if (depth > 4) {
        fprintf (stderr, 
                 "Unsupported number of bits per bixel!\n");
        return 0;
    }
*/
    /* Open destination-file for output */

    fp = fopen (outfile, "wb");
    if (fp == NULL) {
        fprintf (stderr, "Could not open file %s!\n", outfile);
        return 0;
    }

    /* write values as float to file */ 
 
    if (depth > 4) 
    {
        nv = *nvals;
        while (nv > 0)
        {
            nc = nv > LBUFLEN ? LBUFLEN : nv;
            for (i = 0; i < nc; i++)
                fbuf[i] = *vals++;
            fwrite (fbuf, sizeof(float), nc, fp);
            nv -= nc;
        }
        fclose(fp);
        return 1;
    }

    ovals = vals;

    /* get number of rows and columns */

    rldec_get_size (vals, &nrows, &ncols);   

    /* check if the buffer is large enough to hold one uncompressed line */

    if (ncols > LBUFLEN) {
        fprintf (stderr, "ERROR: Number of columns larger than %d!\n",
                 LBUFLEN);
        return 0;
    }

    /* write P5 header for pgm-format */

    if (strstr (outfile, ".pgm") != NULL || strstr (outfile, ".PGM"))
    {
        fprintf (fp, "P5\n%d %d\n%5d\n", ncols, nrows, depth == 1 ? 0xff : 0xffff);
    }

    /* skip number of rows */

    *nvals = 0;
    vals ++;
    (*nvals) ++;

    /* decode line by line */

    for (i = 0; i < nrows; i ++) {
        int n;

        /* decompress line */

        if (depth == 4)
        {
            rldec_decompress_line_float (vals, fbuf, &nc, &nv);
        } else {
            rldec_decompress_line (vals, ibuf, &nc, &nv);
        }

        /* check for correct image size */
        
        if (nc != ncols) {
            fprintf (stderr, "Error in run-length decoding!\n");
            fclose (fp);
            return 0;
        }

        /* increase vals pointer */

        vals += nv;
        (*nvals) += nv;

        /* convert to char */

        if (depth == 1) {
            for (j = 0; j < ncols; j ++)
                cbuf[j] = (unsigned char) ibuf[j];
        } else if (depth == 2){
            for (j = 0; j < ncols; j++) {
                cbuf[j*2] = (unsigned char) ((ibuf[j] >> 8) & 0xff);
                cbuf[j*2+1] = (unsigned char) (ibuf[j] & 0xff);
            }
        }

        /* write to file */
        if (depth == 4)
            n = fwrite (fbuf, 1, ncols * depth, fp);
        else
            n = fwrite (cbuf, 1, ncols * depth, fp);
        if (n != (size_t) ncols * depth) {
               fprintf (stderr, "Write error to file '%s'\n", outfile);
               fclose (fp);
               return 0;
        }
    }

    /* close file */

    fclose (fp);

    assert (*nvals == vals - ovals);
    return 1;
}
/*===========================================================================*/
/** \ingroup rlenc_g

   \brief This function encodes a radar image to BUFR runlength-code 

   This function encodes a radar image in memory to BUFR runlength-code 
   and stores the resulting values into an  array \p vals 
   by a call to \ref bufr_val_to_array. 

   \note In difference to the old \ref rlenc function the
   initial length of \p vals must given in the parameter
   \p nvals in order to prevent \ref bufr_val_to_array from writing to an
   arbitrary position.

   \param[in] img      Array holding the uncompressed radar image.
   \param[in] ncols    Number of columns of the image.
   \param[in] nrows    Number of rows of the image.
   \param[in,out] vals    Float-array holding the coded image.
   \param[in,out] nvals   Number of values in \p vals.

   \return The return-value ist 1 on success, 0 on a fault.

   \see rlenc_from_file, rldec_to_mem, rlenc_compress_line_new
*/

int rlenc_from_mem (unsigned short* img, int nrows, int ncols, varfl* *vals, 
                     int *nvals)

{
    unsigned int ibuf[LBUFLEN];
    int i, j;

    if (img == (unsigned short*) NULL) {
        fprintf (stderr, "Image for rlenc not available!\n");
        return 0;
    }

    /* check if the internal buffer is large enough to hold one uncompressed
       line */

    if (ncols > LBUFLEN) {
        fprintf (stderr, "ERROR: Number of columns larger than %d!\n",
                 LBUFLEN);
        return 0;
    }

    /* output number of rows */

    bufr_val_to_array (vals, (varfl) nrows, nvals);  

    /* compress line by line */

    for (i = 0; i < nrows; i ++) {

        /* get row from memory and convert to int */

        for (j = 0; j < ncols; j ++)
            ibuf[j] = (unsigned int) img[i*ncols+j];
        
        /* compress line to varfl array */

        if (!rlenc_compress_line_new (i, ibuf, ncols, vals, nvals)) {
            return 0;
        }
    }
    return 1;
}


/*===========================================================================*/
/** \ingroup rlenc_g

   \brief This function encodes a radar image to BUFR runlength-code 

   This function encodes a radar image in memory to BUFR runlength-code 
   and stores the resulting values into an  array \p vals 
   by a call to \ref bufr_val_to_array. 

   \note In difference to the old \ref rlenc function the
   initial length of \p vals must given in the parameter
   \p nvals in order to prevent \ref bufr_val_to_array from writing to an
   arbitrary position.

   \param[in] img      Array holding the uncompressed radar image.
   \param[in] ncols    Number of columns of the image.
   \param[in] nrows    Number of rows of the image.
   \param[in,out] vals    Float-array holding the coded image.
   \param[in,out] nvals   Number of values in \p vals.

   \return The return-value ist 1 on success, 0 on a fault.

   \see rlenc_from_file, rldec_to_mem, rlenc_compress_line_new, 
   rlenc_to_mem_float
*/

int rlenc_from_mem_float (float* img, int nrows, int ncols, varfl* *vals, 
                     int *nvals)

{
    float fbuf[LBUFLEN];
    int i;

    if (img == (float*) NULL) {
        fprintf (stderr, "Image for rlenc not available!\n");
        return 0;
    }

    /* check if the internal buffer is large enough to hold one uncompressed
       line */

    if (ncols > LBUFLEN) {
        fprintf (stderr, "ERROR: Number of columns larger than %d!\n",
                 LBUFLEN);
        return 0;
    }

    /* output number of rows */

    bufr_val_to_array (vals, (varfl) nrows, nvals);  

    /* compress line by line */

    for (i = 0; i < nrows; i ++) {

        /* get row from memory */

        memcpy (fbuf, img+i*ncols, ncols * sizeof(float));
        
        /* compress line to varfl array */

        if (!rlenc_compress_line_float (i, fbuf, ncols, vals, nvals)) {
            return 0;
        }
    }
    return 1;
}

/*===========================================================================*/
/** \ingroup rldec_g 
    \brief Decodes a BUFR-runlength-encoded radar image to memory

    This function decodes a BUFR-runlength-encoded radar image stored at
    \p vals. The decoded image is stored in an array \p img[] which will be
    allocated by this function if \p img[] = NULL.
    The memory for the image must be freed by the calling function!

    \param[in] vals      Float-array holding the coded image.
    \param[in,out] img  Destination-array for the radar image.
    \param[out] nvals   Number of \ref varfl values needed for the 
                         compressed radar image.
    \param[out] nrows   Number of lines in image
    \param[out] ncols   Number of pixels per line

    \return The return-value ist 1 on success, 0 on a fault.

    \see rlenc_from_mem, rldec_to_file, rldec_decompress_line
*/

int rldec_to_mem (varfl* vals, unsigned short* *img, int* nvals, int* nrows,
                  int* ncols)

{
    int i, j, nc, nv;
    unsigned int ibuf[LBUFLEN];
    varfl *ovals;

    ovals = vals;

    /* get number of rows and columns */

    rldec_get_size (vals, nrows, ncols);   

    /* Allocate memory for image if necessary */

    if (*img == NULL) {
        *img = (unsigned short*) calloc (*nrows * *ncols, 
                                         sizeof (unsigned short));
        if (*img == NULL) {
            fprintf (stderr, "Could not allacote memory for radar image!\n");
            return 0;
        }
    }

    /* check if the buffer is large enough to hold one uncompressed
       line */

    if (*ncols > LBUFLEN) {
        fprintf (stderr, "ERROR: Number of columns larger than %d!\n",
                 LBUFLEN);
        return 0;
    }

    /* skip number of rows */

    *nvals = 0;
    vals ++;
    (*nvals) ++;

    /* decode line by line */

    for (i = 0; i < *nrows; i ++) {

        /* decompress line */

        rldec_decompress_line (vals, ibuf, &nc, &nv);

        /* check for correct image size */
        
        if (nc != *ncols) {
            fprintf (stderr, "Error in run-length decoding!\n");
            return 0;
        }

        /* increase vals pointer */

        vals += nv;
        (*nvals) += nv;

        /* convert to short and write to memory*/

        for (j = 0; j < *ncols; j ++)
            (*img)[i * *ncols + j] = (unsigned short) ibuf[j];
    }

    assert (*nvals == vals - ovals);
    return 1;
}
/*===========================================================================*/
/** \ingroup rldec_g 
    \brief Decodes a BUFR-runlength-encoded float image to memory

    This function decodes a BUFR-runlength-encoded float image stored at
    \p vals. The decoded image is stored in an array \p img[] which will be
    allocated by this function if \p img[] = NULL.
    The memory for the image must be freed by the calling function!

    \param[in] vals      Float-array holding the coded image.
    \param[in,out] img  Destination-array for the radar image.
    \param[out] nvals   Number of \ref varfl values needed for the 
                         compressed radar image.
    \param[out] nrows   Number of lines in image
    \param[out] ncols   Number of pixels per line

    \return The return-value ist 1 on success, 0 on a fault.

    \see rlenc_from_mem_float, rldec_to_file, rldec_decompress_line_float
*/

int rldec_to_mem_float (varfl* vals, float* *img, int* nvals, int* nrows,
                  int* ncols)

{
    int i, nc, nv;
    float fbuf[LBUFLEN];
    varfl *ovals;

    ovals = vals;

    /* get number of rows and columns */

    rldec_get_size (vals, nrows, ncols);   

    /* Allocate memory for image if necessary */

    if (*img == NULL) {
        *img = (float*) calloc (*nrows * *ncols, 
                                         sizeof (float));
        if (*img == NULL) {
            fprintf (stderr, "Could not allacote memory for radar image!\n");
            return 0;
        }
    }

    /* check if the buffer is large enough to hold one uncompressed
       line */

    if (*ncols > LBUFLEN) {
        fprintf (stderr, "ERROR: Number of columns larger than %d!\n",
                 LBUFLEN);
        return 0;
    }

    /* skip number of rows */

    *nvals = 0;
    vals ++;
    (*nvals) ++;

    /* decode line by line */

    for (i = 0; i < *nrows; i ++) {

        /* decompress line */

        rldec_decompress_line_float (vals, fbuf, &nc, &nv);

        /* check for correct image size */
        
        if (nc != *ncols) {
            fprintf (stderr, "Error in run-length decoding!\n");
            return 0;
        }

        /* increase vals pointer */

        vals += nv;
        (*nvals) += nv;

        /* write to memory*/

        memcpy ((*img)+i * *ncols, fbuf, *ncols * sizeof(float));
    }

    assert (*nvals == vals - ovals);
    return 1;
}

/*===========================================================================*/
/** \ingroup rlenc_g 
    \brief Encodes one line of a radar image to BUFR runlength-code

    This function encodes one line of a radar image to BUFR runlength-code and
    stores the resulting values to array \p dvals by a call to 
    \ref bufr_val_to_array.

    \note In difference to the old \ref rlenc_compress_line function the
    initial length of \p vals must given in the parameter
    \p nvals in order to prevent \ref bufr_val_to_array from writing to an
    arbitrary position.

    \param[in] line     Line number.
    \param[in] src      Is where the uncompressed line is stored.
    \param[in] ncols    Number of pixels per line.
    \param[in,out] dvals    Float-array holding the coded image.
    \param[in,out] nvals    Number of values in VALS.
   
    \return The function returns 1 on success, 0 on a fault.

    \see rldec_decompress_line
*/

int rlenc_compress_line_new (int line, unsigned int* src, int ncols, 
                             varfl* *dvals, int *nvals)

{
    int count, i, n, npar, lens[LBUFLEN], cw, ncgi, nngi = 0;
    unsigned int val, lval = 0, vals[LBUFLEN];
    varfl encbuf[ENCBUFL];

    /* line is runlength-compressed now to N parts, each of them identified 
       by a length (LENS) and a value (VALS). */

    count = n = 0;
    for (i = 0; i < ncols; i ++) {
        val = *(src + i);
    
        /* limit length of one part to 255 to ensure that descriptor 0 31 001 
           does not exceed */
        if (i != 0 && (val != lval || count >= 255)) {  
            lens[n] = count;
            vals[n] = lval;
            n ++;
            count = 0;
            lval = val;
        }
        lval = val;
        count ++;
    }
    lens[n] = count;
    vals[n] = lval;
    n ++;

    /* Count number of parcels. One parcel is identified by a COUNT of 1
       followed by a COUNT > 1 */

    npar = 0;
    for (i = 0; i < n - 1; i ++) if (lens[i] == 1 && lens[i+1] > 1) npar ++;
    npar ++;

    /* output line-number */

    for (i = 0; i < ENCBUFL; i ++) encbuf[i] = (varfl) 0.0;
    cw = 0;
    encbuf[cw++] = line;  

    /* compress it to parcels */
  
    encbuf[cw++] = npar;                   /* number of parcels */

    ncgi = cw ++;                          /* is where the number of 
                                              compressable groups is stored */
    encbuf[ncgi] = (varfl) 0.0;            /* number of compressable groups */

    i = 0;
    for (i = 0; i < n; i ++) {
        if (lens[i] > 1) {                   /* compressable group found */
            if (i > 0 && lens[i-1] == 1) {   /* A new parcel starts here */
                ncgi = cw ++;                /* where the number of compress-
                                                able groups is stored */
                encbuf[ncgi] = (varfl) 0.0;  /* number of compressable groups*/
            }
            encbuf[ncgi] += 1.0;
            encbuf[cw++] = lens[i];
            encbuf[cw++] = vals[i];
        }
        else {                               /* non compressable group found */
            if (i == 0 || lens[i-1] != 1) {  /* this is the first 
                                                uncompressable group in 
                                                the current parcel */
                nngi = cw ++;                /* is where the number of non 
                                               compressable groups is stored */
                encbuf[nngi] = (varfl) 0.0;  /* Number of non compressable 
                                                groups */
            }
            encbuf[nngi] += 1.0;
            encbuf[cw++] = vals[i];
        }
    }
    if (lens[n-1] != 1) encbuf[cw++] = 0;  /* number of noncompressable 
                                              groups in the last parcel = 0 */
    assert (cw <= ENCBUFL);
    
    /* compresson to parcels finished, write values to destination array */

    for (i = 0; i < cw; i ++) {
        if (!bufr_val_to_array (dvals, encbuf[i], nvals)) 
            return 0;
    }

    return 1;
}

/*===========================================================================*/
/** \ingroup rlenc_g 
    \brief Encodes one line of a radar image to BUFR runlength-code

    This function encodes one line of a radar image to BUFR runlength-code and
    stores the resulting values to array \p dvals by a call to 
    \ref bufr_val_to_array.

    \note In difference to the old \ref rlenc_compress_line function the
    initial length of \p vals must given in the parameter
    \p nvals in order to prevent \ref bufr_val_to_array from writing to an
    arbitrary position.

    \param[in] line     Line number.
    \param[in] src      Is where the uncompressed line is stored.
    \param[in] ncols    Number of pixels per line.
    \param[in,out] dvals    Float-array holding the coded image.
    \param[in,out] nvals    Number of values in VALS.
   
    \return The function returns 1 on success, 0 on a fault.

    \see rldec_decompress_line, rldec_decompress_line_float
*/

int rlenc_compress_line_float (int line, float* src, int ncols, 
                             varfl* *dvals, int *nvals)

{
    int count, i, n, npar, lens[LBUFLEN], cw, ncgi, nngi = 0;
    float val, lval = 0, vals[LBUFLEN];
    varfl encbuf[ENCBUFL];

    /* line is runlength-compressed now to N parts, each of them identified 
       by a length (LENS) and a value (VALS). */

    count = n = 0;
    for (i = 0; i < ncols; i ++) {
        val = *(src + i);
    
        /* limit length of one part to 255 to ensure that descriptor 0 31 001 
           does not exceed */
        if (i != 0 && (val != lval || count >= 255)) {  
            lens[n] = count;
            vals[n] = lval;
            n ++;
            count = 0;
            lval = val;
        }
        lval = val;
        count ++;
    }
    lens[n] = count;
    vals[n] = lval;
    n ++;

    /* Count number of parcels. One parcel is identified by a COUNT of 1
       followed by a COUNT > 1 */

    npar = 0;
    for (i = 0; i < n - 1; i ++) if (lens[i] == 1 && lens[i+1] > 1) npar ++;
    npar ++;

    /* output line-number */

    for (i = 0; i < ENCBUFL; i ++) encbuf[i] = (varfl) 0.0;
    cw = 0;
    encbuf[cw++] = line;  

    /* compress it to parcels */
  
    encbuf[cw++] = npar;                   /* number of parcels */

    ncgi = cw ++;                          /* is where the number of 
                                              compressable groups is stored */
    encbuf[ncgi] = (varfl) 0.0;            /* number of compressable groups */

    i = 0;
    for (i = 0; i < n; i ++) {
        if (lens[i] > 1) {                   /* compressable group found */
            if (i > 0 && lens[i-1] == 1) {   /* A new parcel starts here */
                ncgi = cw ++;                /* where the number of compress-
                                                able groups is stored */
                encbuf[ncgi] = (varfl) 0.0;  /* number of compressable groups*/
            }
            encbuf[ncgi] += 1.0;
            encbuf[cw++] = lens[i];
            encbuf[cw++] = vals[i];
        }
        else {                               /* non compressable group found */
            if (i == 0 || lens[i-1] != 1) {  /* this is the first 
                                                uncompressable group in 
                                                the current parcel */
                nngi = cw ++;                /* is where the number of non 
                                               compressable groups is stored */
                encbuf[nngi] = (varfl) 0.0;  /* Number of non compressable 
                                                groups */
            }
            encbuf[nngi] += 1.0;
            encbuf[cw++] = vals[i];
        }
    }
    if (lens[n-1] != 1) encbuf[cw++] = 0;  /* number of noncompressable 
                                              groups in the last parcel = 0 */
    assert (cw <= ENCBUFL);
    
    /* compresson to parcels finished, write values to destination array */

    for (i = 0; i < cw; i ++) {
        if (!bufr_val_to_array (dvals, encbuf[i], nvals)) 
            return 0;
    }

    return 1;
}

/*===========================================================================*/
/** \ingroup rldec_g

    \brief Decodes one line of a float image from BUFR runlength-code

    This function decodes one line of a float image from BUFR runlength-code 
    and stores the resulting values to array \p dest which has to be large
    enough to hold a line.

    \param[in]  vals     Float-array holding the coded image.
    \param[out] dest     Is where the uncompressed line is stored.
    \param[out] ncols    Number of pixels per line.
    \param[out] nvals    Number of values needed for compressed line.

    \see rlenc_compress_line_float
   
*/

void rldec_decompress_line_float (varfl* vals, float* dest, int* ncols, 
                                  int* nvals) 
{
    int i = 0, j, k, l, count = 0, npar, ngr, nup;
    float val;
    varfl* ovals;

    ovals = vals;
    vals ++;                          /* skip linenumber */
    npar = (int) *vals ++;            /* number of parcels */
    for (j = 0; j < npar; j ++) {     /* loop for parcels */
        ngr = (int) *vals ++;         /* number of compressable groups */
        for (k = 0; k < ngr; k ++) {  /* loop for compressable groups */
            count = (int) *vals ++;
            if (*vals == MISSVAL) {
                val = MISSVAL;
                vals ++;
            } else {
                val = *vals ++;
            }
            for (l = 0; l < count; l ++) {  /* loop for length of group */
                dest[i++] = val;
            }
        }
        nup = (int) *vals ++;         /* number of uncompressable pixels */
        for (k = 0; k < nup; k ++) {  /* loop for uncompressable pixels */
            if (*vals == MISSVAL) {
                dest[i++] = MISSVAL;
                vals ++;
            } else {
                dest[i++] = *vals ++;
            }
        }
    }


    *nvals = vals - ovals;
    *ncols = i;
}

/*===========================================================================*/
/** \ingroup rldec_g

    \brief Decodes one line of a radar image from BUFR runlength-code

    This function decodes one line of a radar image from BUFR runlength-code 
    and stores the resulting values to array \p dest which has to be large
    enough to hold a line.

    \param[in]  vals     Float-array holding the coded image.
    \param[out] dest     Is where the uncompressed line is stored.
    \param[out] ncols    Number of pixels per line.
    \param[out] nvals    Number of values needed for compressed line.

    \see rlenc_compress_line_new
   
*/

void rldec_decompress_line (varfl* vals, unsigned int* dest, int* ncols, 
                            int* nvals) 
{
    int i = 0, j, k, l, count = 0, npar, ngr, nup;
    unsigned int val;
    varfl* ovals;

    ovals = vals;
    vals ++;                          /* skip linenumber */
    npar = (int) *vals ++;            /* number of parcels */
    for (j = 0; j < npar; j ++) {     /* loop for parcels */
        ngr = (int) *vals ++;         /* number of compressable groups */
        for (k = 0; k < ngr; k ++) {  /* loop for compressable groups */
            count = (int) *vals ++;
            if (*vals == MISSVAL) {
                val = 0xFFFF;
                vals ++;
            } else {
                val =   (unsigned int) *vals ++;
            }
            for (l = 0; l < count; l ++) {  /* loop for length of group */
                dest[i++] = val;
            }
        }
        nup = (int) *vals ++;         /* number of uncompressable pixels */
        for (k = 0; k < nup; k ++) {  /* loop for uncompressable pixels */
            if (*vals == MISSVAL) {
                dest[i++] = 0xFFFF;
                vals ++;
            } else {
                dest[i++] = (unsigned int) *vals ++;
            }
        }
    }


    *nvals = vals - ovals;
    *ncols = i;
}

/*===========================================================================*/
/** \ingroup rldec_g
    \brief Gets the number of rows and columns of a runlength compressed image

    This function gets the number of rows and colums of a runlength compressed
    image stored at array \p vals 

    \param[in] vals     Float-array holding the coded image.
    \param[out] nrows   Number of lines in image.
    \param[out] ncols   Number of pixels per line.

    \see rldec_to_file, rldec_decompress_line
*/
void rldec_get_size (varfl* vals, int* nrows, int* ncols)
{
    int npar, ngr, nup, j, k, l, count;
    
    *nrows = (int) *vals ++;            /* number of rows */
    *ncols = 0;
    vals ++;                            /* skip linenumber */
    npar = (int) *vals ++;              /* number of parcels */
    for (j = 0; j < npar; j ++) {       /* loop for parcels */
        ngr = (int) *vals ++;           /* number of compressable groups */
        for (k = 0; k < ngr; k ++)  {   /* loop for compressable groups */
            count = (int) *vals ++;
            vals ++;			            /* skip pixel value */
            for (l = 0; l < count; l ++) {  /* loop for length of group */
                (*ncols) ++;	  
            }
        }
        nup = (int) *vals ++;             /* number of uncompressable pixels */
        for (k = 0; k < nup; k ++) {      /* loop for uncompressable pixels */
            vals ++;			          /* skip pixel value */
            (*ncols) ++;
        }
    }
}

/* end of file */



/* bitio */


/*===========================================================================*/
/* functions for bit-io follow:                                   */
/*===========================================================================*/

/* internal data and definitions needed to hold the bitstreams: */

#define MAXIOSTR    10                 /* Max. number of streams that can be 
                                          open simultaneously */
#define INCSIZE     1000               /* for holding a bitstream the open-
                                          function allocates INCSIZE bytes of
                                          memory. When outputing to the 
                                          bitstream and the size of the
                                          memoryblocks exceeds subsequent 
                                          blocks with INCSIZE bytes are
                                          allocated to hold the bitstream. This
                                          is done by a realloc of the buffer. */

typedef struct bitio_stream {          /* structure that defines a bitstrem */
  int used;                            /* identifier if the bitstream is used */
  char *buf;                           /* buffer holding the bitstream */
  long nbits;                          /* currend size of bitstream (counted 
                                          in bits !) */
  size_t size;                         /* current size of allocated memory for
                                          holding the bitstream. */
} bitio_stream;

bitio_stream bios[MAXIOSTR];          /* Data describing MAXIOSTR bitstreams */
int first = 1;                         /* to indicate the first call to one of these functions */

/*===========================================================================*/
/** \ingroup bitio
    \brief This function opens a bitstream for input.

    This function opens a bitstream for input.

    \param[in] buf    Buffer to be used for input
    \param[in] size   Size of buffer.

    \return the function returns a handle by which the bitstream can be 
    identified for all subsequent actions or -1 if the maximum number of 
    opened bitstreams exceeds.

    \see bitio_i_close, bitio_i_input, bitio_o_open
*/

int bitio_i_open (void* buf, size_t size)
{
  int i, handle;

  /* On the first call mark all bitstreams as unused */

  if (first) {
    for (i = 0; i < MAXIOSTR; i ++) bios[i].used = 0;
    first = 0;
  }

  /* search for an unused stream. */

  for (handle = 0; handle < MAXIOSTR; handle ++) {
    if (!bios[handle].used) goto found;
  }
  return -1;

  /* unused bitstream found -> initialize bitstream-data */

found:
  bios[handle].used = 1;
  bios[handle].buf = (char *) buf;
  bios[handle].size = size;
  bios[handle].nbits = 0;                 /* Holds the current bitposition */
  return handle;
}

/*===========================================================================*/
/** \ingroup bitio
    \brief This function reads a value from a bitstream.

    This function reads a value from a bitstream. The bitstream must have 
    been opened by \ref bitio_i_open.

    \param[in] handle   Identifies the bitstream.
    \param[out] val     Is where the input-value is stored.
    \param[in] nbits    Number of bits the value consists of.

    \return Returns 1 on success or 0 on a fault (number of bytes in the
    bitstream exceeded).

    \see bitio_i_open, bitio_i_close, bitio_o_outp
*/

int bitio_i_input (int handle, unsigned long* val, int nbits)
{
  int i, bit;
  size_t byte;
  unsigned long l, bitval;
  char *pc;

  
  l = 0;
  for (i = nbits - 1; i >= 0; i --) {

      /* calculate bit- and byte-number for input and check if bytenumber is
         in a valid range */

      byte = (int) (bios[handle].nbits / 8);
      bit  = (int) (bios[handle].nbits % 8);
      bit = 7 - bit;
      if (byte >= bios[handle].size) return 0;

      /* get bit-value from input-stream */

      pc = bios[handle].buf + byte;
      bitval = (unsigned long) ((*pc >> bit) & 1);

      /* Set a 1-bit in the data value, 0-bits need not to be set, as L has
         been initialized to 0 */

      if (bitval) {
          l |= (bitval << i);
      }
      bios[handle].nbits ++;
  }
  *val = l;
  return 1;
}

/*===========================================================================*/
/** \ingroup bitio
    \brief Closes an bitstream that was opened for input 

    Closes an bitstream that was opened for input 

    \param[in] handle Handle that identifies the bitstream.

    \see bitio_i_open, bitio_i_input
*/

void bitio_i_close (int handle)

{
  bios[handle].used = 0;
}

/*===========================================================================*/
/** \ingroup bitio
    \brief Opens a bitstream for output.

    This function opens a bitstream for output. 

    \return The return-vaule is a handle by which the bit-stream 
    can be identified for all subesquent actions or -1
    if there is no unused bitstream available.
*/

int bitio_o_open ()

{
  int i, handle;

  /* On the first call mark all bitstreams as unused */

  if (first) {
    for (i = 0; i < MAXIOSTR; i ++) bios[i].used = 0;
    first = 0;
  }

  /* search for an unused stream. */

  for (handle = 0; handle < MAXIOSTR; handle ++) {
    if (!bios[handle].used) goto found;
  }
  return -1;

  /* unused bitstream found -> initalize it and allocate memory for it */

found:
  bios[handle].buf = (char *) malloc (INCSIZE);
  if (bios[handle].buf == NULL) return -1;
  memset (bios[handle].buf, 0, INCSIZE);
  bios[handle].used = 1;
  bios[handle].nbits = 0;
  bios[handle].size = INCSIZE;

  return handle;
}

/*===========================================================================*/
/** \ingroup bitio
    \brief This function appends a value to a bitstream.

    This function appends a value to a bitstream which was opened by
    \ref bitio_o_open.

    \param[in] handle  Indicates the bitstream for appending.
    \param[in] val     Value to be output.
    \param[in] nbits   Number of bits of \p val to be output to the stream. 

    \note \p nbits must be less than sizeof (\p long)

    \return The return-value is the bit-position of the value in the 
    bit-stream, or -1 on a fault.

    \see bitio_o_open, bitio_o_close, bitio_o_outp
*/

long bitio_o_append (int handle, unsigned long val, int nbits)

{
    /* Check if bitstream is allready initialized and number of bits does not
       exceed sizeof (unsigned long). */

  assert (bios[handle].used);
  assert (sizeof (unsigned long) * 8 >= nbits);

  /* check if there is enough memory to store the new value. Reallocate
     the memory-block if not */

  if ((bios[handle].nbits + nbits) / 8 + 1 > (long) bios[handle].size) {
    bios[handle].buf = realloc (bios[handle].buf, bios[handle].size + INCSIZE);
    if (bios[handle].buf == NULL) return 0;
	memset (bios[handle].buf + bios[handle].size, 0, INCSIZE);
    bios[handle].size += INCSIZE;
  }

  /* output data to bitstream */

  bitio_o_outp (handle, val, nbits, bios[handle].nbits);
  bios[handle].nbits += nbits;

  return bios[handle].nbits;
}

/*===========================================================================*/
/** \ingroup bitio
    \brief This function outputs a value to a specified position of a bitstream

    This function outputs a value to a specified position of a bitstream.

    \param[in] handle  Indicates the bitstream for output.
    \param[in] val     Value to be output.
    \param[in] nbits   Number of bits of \p val to be output to the stream. 
    \param[in] bitpos  bitposition of the value in the bitstream.

    \note \p nbits must be less then sizeof (\p long)

    \see bitio_o_open, bitio_o_close, bitio_o_append, bitio_i_input

*/

void bitio_o_outp (int handle, unsigned long val, int nbits, long bitpos)

{
  int i, bit, bitval;
  size_t byte;
  char *pc, c;

  /* Check if bitstream is allready initialized and number of bits does not
     exceed sizeof (unsigned long). */

  assert (bios[handle].used);
  assert (sizeof (unsigned long) * 8 >= nbits);

  for (i = nbits - 1; i >= 0; i --) {

      /* Get bit-value */

    bitval = (int) (val >> i) & 1;

    /* calculate bit- and byte-number for output */

    byte = (int) (bitpos / 8);
    bit  = (int) (bitpos % 8);
    bit  = 7 - bit;

    /* set bit-value to output stream */

    pc = bios[handle].buf + byte;
    if (bitval) {
      c = (char) (1 << bit);
      *pc |= c;
    }
    else {
      c = (char) (1 << bit);
      c ^= 0xff;
      *pc &= c;
    }
    bitpos ++;
  }
}

/*===========================================================================*/
/** \ingroup bitio
    \brief Returns the size of an output-bitstream (number of bytes) 

    This function returns the size of an output-bitstream (number of bytes) 
    
    \param[in] handle Identifies the bitstream

    \return Size of the bitstream.

    \see bitio_o_open, bitio_o_outp, bitio_o_append
*/


size_t bitio_o_get_size (int handle)

{
  if (!bios[handle].used) return 0;

  return (size_t) ((bios[handle].nbits - 1) / 8 + 1);
}


/*===========================================================================*/
/** \ingroup bitio
    \brief This function closes an output-bitstream

    This function closes an output-bitstream identified by \p handle and 
    returns a pointer to the memory-area holding the bitstream.

    \param[in] handle   Bit-stream-handle
    \param[out] nbytes  number of bytes in the bitstream.

    \return 
    The funcion returns a pointer to the memory-area holding the bit-stream or
    NULL if an invalid handle was specified. The memory area must be freed by
    the calling function.

    \see bitio_o_open, bitio_o_outp, bitio_o_append, bitio_i_close
*/

void *bitio_o_close (int handle, size_t* nbytes)
{

  if (!bios[handle].used) return NULL;

/* Fill up the last byte with 0-bits */

  while (bios[handle].nbits % 8 != 0) bitio_o_append (handle, 0, 1);

  *nbytes = (size_t) ((bios[handle].nbits - 1) / 8 + 1);
  bios[handle].used = 0;
  return (void *) bios[handle].buf;
}

/* end of file */




/*bufr.c*/
#define BUFR_MAIN


/*===========================================================================*/
/* globals */
/*===========================================================================*/


/*===========================================================================*/
/* default values */
/*===========================================================================*/

/* Define default values for the originating center (OPERA) 
   and the versions of master (WMO) and local (OPERA) table */
#define SUBCENTER 0
#define GENCENTER 247
#define VMTAB 14
#define VLTAB 8

/*===========================================================================*/
/* internal data                                                             */
/*===========================================================================*/

#define MAXREPCOUNT 300      /* Max. replication count */
#define MAX_ADDFIELDS 50     /* Maximum number of nested associated fields */

/* The following variables are used to hold date/time-info of the
   last BUFR-message created. */

static long year_, mon_, day_, hour_, min_;
static int af_[MAX_ADDFIELDS];  /* remember associated fields for nesting */
static int naf_ = 0;            /* current number of associated field */
static int datah_ = -1;          /* bitstream-handle for data-section */
static bufrval_t* vals_ = NULL;  /* structure for holding data values */

static dd cf_spec_des[MAX_ADDFIELDS];    /* remember changed descriptors */
static varfl cf_spec_val[MAX_ADDFIELDS]; /* original referecne values */
static int cf_spec_num = 0;              /* number of changed descriptors */ 
static int ccitt_dw = 0;                 /* data width change 2 8 yyy */
static int incr_scale = 0;               /* scale change 2 7 yyy */

/*===========================================================================*/
/* internal functions                                                        */
/*===========================================================================*/

static int bufr_val_to_datasect (varfl val, int ind);
static int bufr_val_from_datasect (varfl *val, int ind);
static int get_lens (char* buf, long len, int* secl);


/*===========================================================================*/
/* functions */
/*===========================================================================*/

/** \ingroup deprecated_g
    \deprecated use \ref free_descs instead

    This function frees all memory-blocks allocated by \ref read_tables
 */

void bufr_clean (void)


{
  free_descs();
}

/*===========================================================================*/
/** \ingroup deprecated
    \deprecated Use \ref bufr_encode_sections34 instead.

   \brief Creates section 3 and 4 of BUFR message from arrays of data and
  data descriptors.

  This function codes data from an array data descriptors \p descs and an 
  array of varfl-values \p vals to a data section and
  a data descripor section of a BUFR message. Memory for both sections is
  allocated in this function and must be freed by the calling functions.


   \param[in] descs 
     Data-descriptors corresponding to \p vals. 
     For each descriptor there
     must be a data-vaule stored in \p vals. \p descs may also include
     replication factors and sequence descriptors. 
     In that case there must be a larger number of
     \p vals then of \p descs.

   \param[in] ndescs  
     Number of data descriptos contained in \p descs.

   \param[in] vals    
     Data-values to be coded in the data section. For each entry in
     \p descs there must be an entry in \p vals. If there are relication
     factors in \p descs, of course there must be as much \p vals as definded
     by the replication factor.

   \param[out] datasec 
     Is where the data-section (section 4) is stored. The memory-area for the
     data-section is allocated by this function and must be freed by
     the calling function.

   \param[out] ddsec   
     Is where the data-descriptor-section (section 3) in stored. 
     The memory needed is
     allocated by this function and must be freed by the calling 
     function.

   \param[out] datasecl 
     Number of bytes in \p datasec.

   \param[out] ddescl
     Number of bytes in \p ddsec.

   \return The return-value is 1 if data was successfully stored, 0 if not.

   \see bufr_read_msg, bufr_data_from_file

*/


int bufr_create_msg (dd* descs, int ndescs, varfl* vals, void **datasec, 
                     void **ddsec, size_t *datasecl, size_t *ddescl)


{
    bufrval_t* valarray = NULL;
    int ok, desch;
    bufr_t msg;

    memset (&msg, 0, sizeof (bufr_t));

    year_ = mon_ = day_ = hour_ = min_ = 0;

    /* Open two bitstreams, one for data-descriptors, one for data */

    desch = bufr_open_descsec_w (1);

    ok = (desch >= 0);

    if (ok)
        ok = (bufr_open_datasect_w () >= 0);

    /* output data to the data descriptor bitstream */

    if (ok)
        bufr_out_descsec (descs, ndescs, desch);

    /* set global array */

    if (ok) {
        valarray = bufr_open_val_array ();
        ok = (valarray != (bufrval_t*) NULL);
    }

    if (ok) {
        valarray->vals = vals;
        valarray->vali = 0;
    }

    /* output data to the data-section */

    if (ok) {
        ok = bufr_parse_in (descs, 0, ndescs - 1, bufr_val_from_global, 0);
        valarray->vals = (varfl*) NULL;
        bufr_close_val_array ();
    }

    /* close bitstreams and write data to bufr message */

    bufr_close_descsec_w (&msg, desch);

    *ddsec = msg.sec[3];
    *ddescl = (size_t) msg.secl[3];

    bufr_close_datasect_w (&msg);

    *datasec = msg.sec[4];
    *datasecl = (size_t) msg.secl[4];
    
    return ok;
}
/*===========================================================================*/
/** \ingroup basicin
   \brief Creates section 3 and 4 of BUFR message from arrays of data and
  data descriptors.

  This function codes data from an array data descriptors \p descs and an 
  array of varfl-values \p vals to a data section and
  a data descripor section of a BUFR message. Memory for both sections is
  allocated in this function and must be freed by the calling functions.


   \param[in] descs
     Data-descriptors corresponding to \p vals. 
     For each descriptor there
     must be a data-vaule stored in \p vals. \p descs may also include
     replication factors and sequence descriptors. 
     In that case there must be a larger number of
     \p vals then of \p descs.

   \param[in] ndescs  
     Number of data descriptos contained in \p descs.

   \param[in] vals    
     Data-values to be coded in the data section. For each entry in
     \p descs there must be an entry in \p vals. If there are relication
     factors in \p descs, of course there must be as much \p vals as definded
     by the replication factor.

   \param[out] msg The BUFR message where to store the coded descriptor and
                   data sections. The memory-area for both sections
                   is allocated by this function and must be freed by
                   the calling function using \ref bufr_free_data.

   \return The return-value is 1 if data was successfully stored, 0 if not.

   \see bufr_encode_sections0125, bufr_data_from_file, bufr_read_msg

*/


int bufr_encode_sections34 (dd* descs, int ndescs, varfl* vals, bufr_t* msg)


{
    char *datasec = NULL; 
    char *ddsec = NULL;
    size_t datasecl = 0;
    size_t ddescl = 0;
    int ret;

    if (msg == (bufr_t*) NULL) {
        fprintf (stderr, "Error writing data to BUFR message\n");
        return 0;
    }
    
    ret = bufr_create_msg (descs, ndescs, vals, (void**) &datasec, 
                           (void**) &ddsec, &datasecl, &ddescl);

    msg->sec[3] = ddsec;
    msg->sec[4] = datasec;
    msg->secl[3] = ddescl;
    msg->secl[4] = datasecl;

    return ret;
}

/*===========================================================================*/
/** \ingroup basicout
    \brief This functions reads the encoded BUFR-message to a binary file

   This function reads the encoded BUFR message from a binary file,
   calculates the section length and writes each section to a memory
   block.
   Memory for the sections is allocated by this function and must be
   freed by the calling function using \ref bufr_free_data.

   \param[in] msg  The complete BUFR message
   \param[in] file The filename of the binary file

   \return 1 on success, 0 on error

   \see bufr_write_file 
*/

int bufr_read_file (bufr_t* msg, const char* file) {

    FILE* fp;           /* file pointer to bufr file */
    char* bm;           /* pointer to memory holding bufr file */
    int len;

    /* open file */

    fp = fopen (file, "rb");
    if (fp == NULL) {
        fprintf (stderr, "unable to open file '%s'\n", file);
        return 0;
    }

    /* get length of message */

    fseek (fp, 0L, SEEK_END);
    len = ftell (fp);
    fseek (fp, 0L, SEEK_SET);

    /* allocate memory and read message */

    bm = (char *) malloc ((size_t) len);
    if (bm == NULL) {
        fprintf (stderr, 
                 "unable to allocate %d bytes to hold BUFR-message !\n", len);
        fclose (fp);
        return 0;
    }
    if (fread (bm, 1, (size_t) len, fp) != (size_t) len) {
        fprintf (stderr, "Error reading BUFR message from file!\n");
        fclose (fp);
        free (bm);
        return 0;
    }

    fclose (fp);

    /* get raw bufr data */

    if (!bufr_get_sections (bm, len, msg)) {
        free (bm);
        return 0;
    }

    free (bm);
    return 1;
}
/*===========================================================================*/
/** \ingroup basicout
    \brief Calculates the section length of a BUFR message and allocates
    memory for each section.

    This function calculates the sections length of a BUFR message
    and allocates memory for each section. 
    The memory has to be freed by the calling function using 
    \ref bufr_free_data.

    \param[in] bm Pointer to the memory where the raw BUFR message is stored
    \param[in] len Length of \p bm
    \param[in,out] msg The BUFR message containing the single sections and
                       section length

    \return Returns the length of the complete BUFR message or 0 on error.

    \see bufr_free_data, bufr_read_file
*/

int bufr_get_sections (char* bm, int len, bufr_t* msg) 

{
    int co, l;
    char* buf;          /* pointer to beginning of BUFR message */
    char* b7777;        /* pointer to end of BUFR message */
    int i;

    /* Search for "BUFR" */

    buf = NULL;
    for (l = 0; l < len - 4 && buf == NULL; l ++) {
        if (*(bm + l)     == 'B' && 
            *(bm + l + 1) == 'U' &&
            *(bm + l + 2) == 'F' &&
            *(bm + l + 3) == 'R') buf = bm + l;
    }
    if (buf == NULL) {
        fprintf (stderr, "'BUFR' not found in BUFR-message !\n");
        return 0;
    }

    /* Check for the ending "7777" */

    b7777 = NULL;
    for (l = 0; l < len - 3 && b7777 == NULL; l ++) {
        if (*(bm + l)     == '7' && 
            *(bm + l + 1) == '7' &&
            *(bm + l + 2) == '7' &&
            *(bm + l + 3) == '7') b7777 = bm + l;
    }
    if (b7777 == NULL) {
        fprintf (stderr, "'7777' not found in BUFR-message !\n");
        return 0;
    }

    /* Get length of all 6 sections */
    
    if (!get_lens (buf, len, msg->secl)) {
        fprintf (stderr, "unable to read lengths of BUFR-sections !\n");
        return 0;
    }

    /* allocate memory for each section */

    co = 0;
    for (i = 0; i < 6; i ++) {
        msg->sec[i] = (char *) malloc ((size_t) msg->secl[i] + 1);
        if (msg->sec[i] == NULL) {
            fprintf (stderr, 
                     "unable to allocate %d bytes for section %d !\n", 
                     msg->secl[i], i);
            return 0;
        }
        memcpy (msg->sec[i], buf + co, (size_t) msg->secl[i]);
        co += msg->secl[i];
    }
    return co;
}

/*===========================================================================*/
/** \ingroup extin
    \brief Write descriptor section of a BUFR message to the bitsream 

    This function writes the descriptor section of a BUFR message 
    to the section 3 bitstream which has already been opened using
    \ref bufr_open_descsec_w

    \param[in]     descp  Array holding the data descriptors
    \param[in]     ndescs Number of descriptors
    \param[in]     desch  Handle to the bitstream

    \return 1 on success, 0 on error

    \see bufr_open_descsec_w, bufr_out_descsec
*/

int bufr_out_descsec (dd *descp, int ndescs, int desch)

{
    unsigned long l;
    int i;

    /* Append data descriptor to data descriptor section */

    for (i = 0; i < ndescs; i ++) {
        l = (unsigned long) descp->f;
        if (!bitio_o_append (desch, l, 2)) return 0;
        l = (unsigned long) descp->x;
        if (!bitio_o_append (desch, l, 6)) return 0;
        l = (unsigned long) descp->y;
        if (!bitio_o_append (desch, l, 8)) return 0;
        descp ++;
    }

    return 1;
}
/*===========================================================================*/
/** \ingroup extin
    \brief Open bitstream for section 3 for writing and set default values 

    This function opens the bitstream for section 3 and sets default values.
    The bistream must be closed using \ref bufr_close_descsec_w.

    \return Returns handle for the bitstream or -1 on error.

    \see bufr_close_descsec_w, bufr_out_descsec
*/

int bufr_open_descsec_w (int subsets) 
{

    size_t n;
    int desch;

    /* open bitstream */

    desch = bitio_o_open ();
    if (desch == -1) {
        bitio_o_close (desch, &n);
        return -1;
    }

    /* output default data */

    bitio_o_append (desch, 0L, 24);  /* length of descriptor-section, set to 
                                         0. The correct length is set by 
                                         close_descsec_w. */
    bitio_o_append (desch, 0L, 8);   /* reserved octet, set to 0 */
    bitio_o_append (desch, subsets, 16);  /* number of data subsets */
    bitio_o_append (desch, 128L, 8); /* observed non-compressed data */
    return desch;
}
/*===========================================================================*/

/** \ingroup extin
    \brief Write length of section 3 and close bitstream

    This function calculates and writes the length of section 3, then closes
    the bitstream.

   \param[in,out] bufr BUFR message to hold the section.
   \param[in]     desch Handle to the bitstream

   \see bufr_open_descsec_w, bufr_out_descsec

*/

void bufr_close_descsec_w(bufr_t* bufr, int desch) {

    int n;
    size_t st;

    if (desch == -1 || bufr == (bufr_t*) NULL) return;

    /* get current length */

    n = (int)  bitio_o_get_size (desch);

    /* number of bytes must be an even number */

	if (n % 2 != 0) bitio_o_append (desch, 0L, 8);

    /* write length of section to beginning */
    
    n = (int) bitio_o_get_size (desch);
    bitio_o_outp (desch, (long) n, 24, 0L);

    /* close bitstream and return pointer */

    bufr->sec[3] = (char *) bitio_o_close (desch, &st);
    bufr->secl[3] = (int) st;
}


/*===========================================================================*/
/** \ingroup deprecated_g
   \deprecated use \ref bufr_encode_sections0125 instead

   Sets up section 0,1,2,5 in a rather easy fashion and takes Section 1 data
   from structure s1.

   \param[in,out]  sec  Sections 0 - 5
   \param[in,out]  secl Lengths of sections 0 - 5
   \param[in]      s1   Data to be put into Section 1
*/
int setup_sec0125 (char* sec[], size_t secl[], sect_1_t s1)

{
    bufr_t msg;
    int i;

    for (i = 0; i < 6; i++) {
        msg.secl[i] = (int) secl[i];
        msg.sec[i] = sec[i];
    }

    if (!bufr_encode_sections0125 (&s1, &msg))
        return 0;

    for (i = 0; i < 6; i++) {
        secl[i] = (size_t) msg.secl[i];
        sec[i]  = msg.sec[i];
    }

    return 1;

}
/*===========================================================================*/
/** \ingroup deprecated_g
    \deprecated Use \ref bufr_write_file instead.

    Write BUFR message to a binary file.

    \param[in] sec     Poiter-Array to the 6 sections.
    \param[in] secl    Length of the sections.
    \param[in] buffile Output-File

    \return The function returns 1 on success, 0 on a fault.
*/

int save_sections (char** sec, size_t* secl, char* buffile)

{
    FILE *fp;
    int i;

    /* open file */

    fp = fopen (buffile, "wb");
    if (fp == NULL) {
        fprintf (stderr, "Could not open file %s!\n", buffile);
        return 0;
    }

    /* output all sections */

    for (i = 0; i < 6; i ++) {
        if (fwrite (sec[i], 1, secl[i], fp) != secl[i]) {
            fclose (fp);
            fprintf (stderr, 
                     "An error occoured writing '%s'. File is invalid !\n", 
                     buffile);
            return 0;
        }
    }

    /* close file and return */

    fclose (fp);
    return 1;
}


/*===========================================================================*/

/** \ingroup utils_g
    \brief Parse data descriptors and call user defined functions for 
    each data element or for each descriptor

   This function, a more general version of \ref bufr_parse, parses 
   a descriptor or a sequence of descriptors and calls the user defined 
   functions \p inputfkt and \p outputfkt for each data-value 
   corresponding to an element descriptor.
   In case of CCITT (ASCII) data it calls the user-functions for each 
   character of the string.

   Data values are read in using the user-defined function \p inputfkt and
   written out using \p outputfkt.

   Optionally the user-defined functions are called for all descriptors, 
   including sequence descriptors and data modification descriptors.

   \param[in] descs      Pointer to the data-descriptors.
   \param[in] start      First data-descriptor for output.
   \param[in] end        Last data-descriptor for output.
   \param[in] inputfkt   User defined input function to be called for each 
                         data-element or descriptor 
   \param[in] outputfkt  User defined ouput function to be called for each 
                         data-element or descriptor
   \param[in] callback_all_descs Flag that indictes when the user-functions
                         are to be called: \n 
                         \b 0 for normal behaviour 
                         (call user-functions for each
                         element descriptor and each CCITT character) \n
                         \b 1 for extended behaviour (call both user-functions
                         also for sequence descriptors and 
                         CCITT descriptors, \n
                         call \p outputfkt also for replication descriptors
                         and data modification descriptors.)

   \return
   The function returns 1 on success, 0 on error.

   \see bufr_parse, bufr_parse_in, bufr_parse_out, \ref cbin,
   \ref cbout

*/

int bufr_parse_new (dd *descs, int start, int end, 
                    int (*inputfkt) (varfl *val, int ind),
                    int (*outputfkt) (varfl val, int ind),
                    int callback_all_descs) {

    int i, j, nrep, nd;
    int ind;                    /* current descriptor index */
    varfl d;                    /* one float value to process */
    dd descr;                   /* current descriptor */
    static int level = 0;       /* recursion level */
    char* tmp;
    int operator_qual;          /* flag that indicates data descriptor
                                   operator qualifiers (0 31 y) */


    /* increase recursion level */

    level ++;

    /* parse all descriptors */

    for (ind = start; ind <= end; ind++) {

        /* get current descriptor */

        memcpy (&descr, descs + ind, sizeof (dd));

        if (descr.f == 0) {

            /* descriptor is element descriptor */

            if ((i = get_index (ELDESC, &descr)) < 0) {

                /* invalid descriptor */

                fprintf (stderr, 
                       "Unknown data descriptor found: F=%d, X=%d, Y=%d !\n", 
                         descr.f, descr.x, descr.y);
                return 0;
            }

            /* Special Treatment for ASCII data */

            if (strcmp (des[i]->el->unit, "CCITT IA5") == 0) {
                
                /* call outputfkt to ouput descriptor and allow
                   use of proper callback for ascii */
                
                if (callback_all_descs) {
                    varfl v;
                    des[_desc_special]->el->d.f = descr.f;
                    des[_desc_special]->el->d.x = descr.x;
                    des[_desc_special]->el->d.y = descr.y;
                    des[_desc_special]->el->dw = des[i]->el->dw;
                    if (ccitt_dw > 0)
                        des[_desc_special]->el->dw = 8 * ccitt_dw;

                    tmp = des[_desc_special]->el->unit;
                    des[_desc_special]->el->unit = des[i]->el->unit;
                    if (!(*inputfkt) (&v, _desc_special)) return 0;
                    if (!(*outputfkt) (0, _desc_special)) return 0;
                    des[_desc_special]->el->unit = tmp;
                    continue;
                }

                /*loop through all bytes of the character 
                  string and store them using the special descriptor 
                  we have created. */

                nrep = des[i]->el->dw / 8;
                if (ccitt_dw > 0)
                    nrep = ccitt_dw;
                    
                for (j = 0; j < nrep; j ++) { 

                    if (!(*inputfkt) (&d, ccitt_special)) return 0;
                    if (!(*outputfkt) (d, ccitt_special)) return 0;
                }
                continue;
            }

            /* Write data to output function. If an "Add associated field" 
               has been set we have to store additional items, 
               except it is a 0 31 y descritor */

            if (_bufr_edition < 3) {
                operator_qual = (des[i]->el->d.x == 31 && 
                                 des[i]->el->d.y == 21);
            } else {
                operator_qual = des[i]->el->d.x == 31;
            }

            if (addfields != 0 && !operator_qual) {
                    
                /* set special descriptor */
                
                des[add_f_special]->el->scale  = 0;
                des[add_f_special]->el->refval = 0;
                des[add_f_special]->el->dw     = addfields;

                /* process data */

                if (!(*inputfkt) (&d, add_f_special)) return 0;
                if (!(*outputfkt) (d, add_f_special)) return 0;
            }

            /* finally process data for the given descriptor */
                
            if (!(*inputfkt) (&d, i)) return 0;
            if (!(*outputfkt) (d, i)) return 0;

            /* Check if this is date/time info and keep this data for 
               further requests in bufr_get_date_time */

            if (descr.x == 4) switch (descr.y)
                {
                case 1: 
                    if (_bufr_edition >= 4) {
                        year_ = (long) d; 
                    }
                    else {
                        year_ = (long) ((int) (d-1) %100 + 1);
                    }
                    break;
                case 2: mon_  = (long) d; break;
                case 3: day_  = (long) d; break;
                case 4: hour_ = (long) d; break;
                case 5: min_  = (long) d; break;
                }
            continue;
        } /* end if (... ELDESC ...) */

        else if (descr.f == 3) {

            /* If data-descriptor is a sequence descriptor -> call this 
               function again for each entry in the sequence descriptor 
               or call user defined callback if parse_seqdescs is not set 
            */

            if ((i = get_index (SEQDESC, &descr)) < 0) {

                /* invalid descriptor */

                fprintf (stderr, 
                       "Unknown data descriptor found: F=%d, X=%d, Y=%d !\n", 
                         descr.f, descr.x, descr.y);
                return 0;
            }

            if (!callback_all_descs) {
                if (!bufr_parse_new (des[i]->seq->del, 0, 
                                     des[i]->seq->nel - 1,
                                     inputfkt, outputfkt, 0)) {
                    return 0;
                }
            }
            else {
                if (!inputfkt (&d, i)) return 0;
                if (!outputfkt (0, i)) return 0;
            }

            continue;
        }

        else if (descr.f == 1) {

            /* replication descriptor */

            nd   = descr.x;
            nrep = descr.y;

            /* output descriptor if not in input mode */
            
            if (callback_all_descs) {

                des[_desc_special]->el->d.f = descr.f;
                des[_desc_special]->el->d.x = descr.x;
                des[_desc_special]->el->d.y = descr.y;
                if (!(*outputfkt) (0, _desc_special)) return 0;
            }

            /* if there is a delayed replication factor */

            if (nrep == 0) {

                /* get number of replications, remember it and write it out*/

                ind++;
                memcpy (&descr, descs + ind, sizeof (dd));
                if ((i = get_index (ELDESC, &descr)) < 0) {
                    fprintf (stderr, 
                        "Unknown data descriptor found: F=%d, X=%d, Y=%d !\n", 
                             descr.f, descr.x, descr.y);
                    return 0;
                }
                if (!(*inputfkt) (&d, i)) return 0;
                nrep = (int) d;
                if (!(*outputfkt) (nrep, i)) return 0;
                
                /* data replication */
                
                if (descr.y == 11 || descr.y == 12)
                    nrep = 1;
            }
            
            /* do the replication now */

            for (i = 0; i < nrep; i ++) {
                if (!bufr_parse_new (descs, ind + 1, ind + nd, inputfkt, 
                                     outputfkt, callback_all_descs))
                    return 0;
                _replicating++;
            }
            _replicating -= nrep;
            ind += nd;
            continue;
        }

        else if (descr.f == 2) {

            /* data modification descriptor */

            if (callback_all_descs) {
            
                /* special treatment for ascii data (2 5 y) */

                if (descr.x == 5)
                {
                    varfl v;
                    des[_desc_special]->el->d.f = descr.f;
                    des[_desc_special]->el->d.x = descr.x;
                    des[_desc_special]->el->d.y = descr.y;
                    des[_desc_special]->el->dw = descr.y * 8;
                    tmp = des[_desc_special]->el->unit;
                    des[_desc_special]->el->unit = "CCITT IA5";
                    if (!(*inputfkt) (&v, _desc_special)) return 0;
                    if (!(*outputfkt) (0, _desc_special)) return 0;
                    des[_desc_special]->el->unit = tmp;
                    continue;
                }
                des[_desc_special]->el->d.f = descr.f;
                des[_desc_special]->el->d.x = descr.x;
                des[_desc_special]->el->d.y = descr.y;
                if (!(*outputfkt) (0, _desc_special)) return 0;
            }

            switch (descr.x) {

                /* change of datawidth, valid until cancelled by 2 01 000 */
            case 1:   
                if (descr.y == 0) {
                    dw = 128;
                } else {
                    dw = descr.y; 
                }
                continue;
                
                /* change of scale, valid until cancelled by 2 02 000 */
            case 2:
                if (descr.y == 0) {
                    sc = 128;
                } else {
                    sc = descr.y;
                }
                continue;

                /* modyify reference values */
            case 3:

                /* revert all reference value  changes */
                
                if (descr.y == 0)
                {
                    while (cf_spec_num--)
                    {
                        i = get_index (ELDESC, &cf_spec_des[cf_spec_num]);
                        des[i]->el->refval = cf_spec_val[cf_spec_num];
                    }
                }
     
                /* stop reference value change */
                
                else if (descr.y == 255)
                    ;
                     
                /* start reference value change */
                
                else
                {
                    des[cf_special]->el->dw = descr.y;
                    des[cf_special]->el->scale = 0;
                    des[cf_special]->el->refval = 0;
                    
                    /* read new ref. value for all following element descriptors, 
                       until 2 3 255 */
                    
                    ind++;
                    while (ind <= end && ! 
                           (descs[ind].f == 2 && descs[ind].x == 3 && descs[ind].y == 255))
                    {
                        memcpy (&descr, descs + ind, sizeof (dd));
                        if ((i = get_index (ELDESC, &descr)) < 0) {
                            fprintf (stderr, 
                            "Unknown data descriptor found: F=%d, X=%d, Y=%d !\n", 
                                 descr.f, descr.x, descr.y);
                            return 0;
                        }
                        
                        /* get new reference value */

                        des[cf_special]->el->d = descr;
                        if (!(*inputfkt) (&d, cf_special)) return 0;
                        if (!(*outputfkt) (d, cf_special)) return 0;

                        /* save old reference value */
                        
                        if (cf_spec_num < MAX_ADDFIELDS)
                        {
                            cf_spec_des[cf_spec_num] = des[i]->el->d;
                            cf_spec_val[cf_spec_num++] = des[i]->el->refval;
                            des[i]->el->refval = d;
                        }
                        else
                        {
                            fprintf (stderr, 
                                "Maximum number of reference value changes!\n");
                            return 0;
                        }
                        ind++;
                    }
                    
                    /* to allow output stop of ref. value 2 3 255 */

                    if (ind <= end)
                        ind--;
                }
                continue;
 
                /* add associated field, valid until canceled by 2 04 000 */
            case 4:
                if (descr.y == 0) {
                    naf_ --;
                    if (naf_ < 0) {
                        fprintf (stderr, "Illegal call of 2 04 000!\n");
                        return 0;
                    }
                    addfields = af_[naf_];
                }
                else {
                    af_[naf_] = addfields;
                    naf_ ++;
                    if (naf_ > MAX_ADDFIELDS) {
                        fprintf (stderr, 
                            "Maximum number of associated fields reached!\n");
                        return 0;
                    }
                    addfields += descr.y;
                }
                continue;

               
            /* signify character */
            case 5: 
                for (i = 0; i < descr.y; i++) 
                { 
                    if (!(*inputfkt) (&d, ccitt_special)) return 0;
                    if (!(*outputfkt) (d, ccitt_special)) return 0;
                }
                continue;

            case 6: /* signify dw for local desc. */
                if (ind < end && get_index (ELDESC, descs + ind + 1) == -1)
                {
                    ind++;
                    des[cf_special]->el->d = descs[ind];
                    des[cf_special]->el->dw = descr.y;
                    des[cf_special]->el->scale = 0;
                    des[cf_special]->el->refval = 0;
                    if (!(*inputfkt) (&d, cf_special)) return 0;
                    if (!(*outputfkt) (d, cf_special)) return 0;
                }
                continue;
                
            case 21: /* data not present */
            case 22: /* quality info follows */
            case 23: /* substituted values op */
            case 24: /* statistical values */
            case 25: /* statistical values */
            case 32: /* replaced values */
            case 35: /* cancel back reference */
            case 36: /* define data present */
            case 37: /* use data present */
                /* these descriptors don't require special en-/decoding */
                continue;

            /* BUFR edition 4 only */
            case 7:  /* increase scale, ref. and width */
                incr_scale = descr.y;
                continue;
                
            case 8: /* change width of CCITT field */
                ccitt_dw = descr.y;
                continue;
                
            case 41: /* event */
            case 42: /* conditioning event */
            case 43: /* categorical forecast */
                /* these descriptors don't require special en-/decoding */
                continue;

                /* invalid descriptor */
            default:
                fprintf (stderr, 
                        "Unknown data modification descriptor found: F=%d, X=%d, Y=%d !\n", 
                         descr.f, descr.x, descr.y);
                return 0;
            }
        }
        else {
            
            /* invalid descriptor */
            
            fprintf (stderr, 
                     "Unknown data descriptor found: F=%d, X=%d, Y=%d !\n", 
                     descr.f, descr.x, descr.y);
            return 0;
        }
        
    } /* end for loop over all descriptors */

    /* decrease recursing level */

    level --;
    return 1;

}

/*===========================================================================*/
/** \ingroup utils_g
    \brief Parse data descriptors and call user-function for each element

   This function parses a descriptor or a sequence of
   descriptors and calls the user defined function
   \p userfkt for each data-value corresponding to an element descriptor.
   In case of CCITT (ASCII) data it calls \p userfkt for each character of
   the string.

   Data values are read from an array of floats stored at \p vals.
   
   \param[in] descs      Pointer to the data-descriptors.
   \param[in] start      First data-descriptor for output.
   \param[in] end        Last data-descriptor for output.
   \param[in] vals       Pointer to an array of values.
   \param[in,out] vali   Index for the array \p vals that identifies the 
                         values to be used for output. 
                         \p vali is increased after data-output.
   \param[in] userfkt    User-function to be called for each data-element

   \return
   The function returns 1 on success, 0 if there was an error outputing to the
   bitstreams.
*/


int bufr_parse (dd* descs, int start, int end, varfl *vals, unsigned *vali,
                int (*userfkt) (varfl val, int ind)) {
    int ok;
    bufrval_t* bufrvals;

    bufrvals = bufr_open_val_array ();

    if (bufrvals == (bufrval_t*) NULL) {
        return 0;
    }

    bufrvals->vals = vals;
    bufrvals->vali = *vali;
    ok = bufr_parse_new (descs, start, end, bufr_val_from_global, userfkt, 
                         0);
    *vali = bufrvals->vali;

    bufrvals->vals = (varfl*) NULL;
    bufr_close_val_array ();
    return ok;
}


/*===========================================================================*/
/** \ingroup extin
    \brief Parse data descriptors and call user defined input function for 
    each element or for each descriptor

   This function, derived from \ref bufr_parse_new, parses 
   a descriptor or a sequence of descriptors and calls the user defined 
   function \p inputfkt for reading each data-value corresponding to an 
   element descriptor.
   In case of CCITT (ASCII) data it calls the user-function for each 
   character of the string.

   Data values are wrote out to the global data section bitstream
   (see \ref bufr_open_datasect_w).

   Optionally \p inputfkt is called also for sequence descriptors 
   and ccitt descriptors

   \param[in] descs      Pointer to the data-descriptors.
   \param[in] start      First data-descriptor for output.
   \param[in] end        Last data-descriptor for output.
   \param[in] inputfkt   User defined input function to be called for each 
                         data-element or descriptor 
   \param[in] callback_descs Flag that indictes when the user-functions
                         are to be called: \n
                         \b 0 for normal behaviour 
                         (call \p inputfkt for each
                         element descriptor and each CCITT character) \n
                         \b 1 for extended behaviour 
                         (call \p inputfkt also
                         for sequence descriptors and CCITT descriptors)
   \return
   The function returns 1 on success, 0 on error

   \see bufr_parse, bufr_parse_new, bufr_parse_in, \ref cbin,
   bufr_open_datasect_w
*/


int bufr_parse_in  (dd *descs, int start, int end, 
                    int (*inputfkt) (varfl *val, int ind),
                    int callback_descs) {

  return bufr_parse_new (descs, start, end, inputfkt,  
                         bufr_val_to_datasect, callback_descs); 
}

/*===========================================================================*/
/** \ingroup extout
    \brief Parse data descriptors and call user defined output function for 
    each element or for each descriptor

   This function, derived from \ref bufr_parse_new, parses 
   a descriptor or a sequence of descriptors and calls the user defined 
   function \p outputfkt for each data-value corresponding to an 
   element descriptor.
   In case of CCITT (ASCII) data it calls the user-function for each 
   character of the string.

   Data values are read from the global data section bitstream
   (see \ref bufr_open_datasect_r).

   Optionally \p outputfkt is called for all descriptors 
   including sequence descriptors, repetition descriptors, ...

   \param[in] descs      Pointer to the data-descriptors.
   \param[in] start      First data-descriptor for output.
   \param[in] end        Last data-descriptor for output.
   \param[in] outputfkt  User defined output function to be called for each 
                         data-element or descriptor 
   \param[in] callback_all_descs Flag that indictes when the user-functions
                         are to be called: \n
                         \b 0 for normal behaviour 
                         (call \p outputfkt for each
                         element descriptor and each CCITT character) \n
                         \b 1 for extended behaviour 
                         (call \p outputfkt for all descriptors)
   \return
   The function returns 1 on success, 0 on error

   \see bufr_parse, bufr_parse_new, bufr_parse_in, \ref cbout,
   bufr_open_datasect_r
*/


int bufr_parse_out  (dd *descs, int start, int end, 
                     int (*outputfkt) (varfl val, int ind),
                     int callback_all_descs) {

    return bufr_parse_new (descs, start, end, bufr_val_from_datasect,  
                           outputfkt, callback_all_descs); 
}


/*===========================================================================*/
/** \ingroup extin
    \brief Reads section 1 from a file and stores data read in s1

    This function reads section 1 from an ASCII file and stores the data
    read in a structure \p s1 .
    If the file can not be read, \p s1 is filled with internally defined 
    default values.

    \param[in,out] s1     Structure where section 1 data is stored.
    \param[in]     file   Filename of the input file.

    \see bufr_sect_1_to_file
*/

void bufr_sect_1_from_file (sect_1_t* s1, const char* file)
{
  FILE *fp;
  char buf[200];
  int val, count;

  /* Set section 1 to default vales */

  s1->mtab    = 0;
  s1->subcent  = SUBCENTER;
  s1->gencent  = GENCENTER;
  s1->updsequ = 0;
  s1->opsec   = 0;
  s1->dcat    = 6;
  s1->idcatst = 0;
  s1->dcatst  = 0;
  s1->vmtab   = VMTAB;
  s1->vltab   = VLTAB;
  s1->year    = 999;
  s1->mon     = 999;
  s1->day     = 999;
  s1->hour    = 999;
  s1->min     = 999;
  s1->sec     = 0;

/* open file and read data */

  fp = fopen (file, "r");
  if (fp == NULL) {
      return;
  }

  count = 0;
  while (fgets (buf, 200, fp) != NULL) {
    if (sscanf (buf, "%d", &val) == 1) {
        switch (count) {
        case 0:  s1->mtab    = val; break;
        case 1:  s1->subcent  = val; break;
        case 2:  s1->gencent  = val; break;
        case 3:  s1->updsequ = val; break;
        case 4:  s1->opsec   = val; break;
        case 5:  s1->dcat    = val; break;
        case 6:  s1->dcatst  = val; break;
        case 7:  s1->vmtab   = val; break;
        case 8:  s1->vltab   = val; break;
        case 9:  s1->year    = val; break;
        case 10:  s1->mon     = val; break;
        case 11: s1->day     = val; break;
        case 12: s1->hour    = val; break;
        case 13: s1->min     = val; break;
            /* new fields for edition 4 */
        case 14: s1->sec     = val; break;
        case 15: s1->idcatst = val; break;
        }
        count ++;
    }
  }
  fclose (fp);
}

/*===========================================================================*/
/** \ingroup basicin
    \brief This function creates sections 0, 1, 2 and 5.

    This function creates sections 0, 1, 2 and 5 of a BUFR message.
    Memory for this section is allocated by this function and must be
    freed by the calling function using \ref bufr_free_data. \n
    The total length of the message is calculeted out of the single
    section length, thus sections 3 and 4 must already be present in
    the bufr message when calling this function.
    The BUFR edition is wrote into section 0 and is taken from the global
    \ref _bufr_edition parameter. \n
    If section 1 data and time parameters are set to 999 (no value), the
    current system time is taken for coding date and time information.

    \param[in] s1 \ref sect_1_t structure containing section 1 data
    \param[in,out] msg BUFR message where the sections are to be stored. Must
                       already contain section 3 and 4.

    \return 1 on success, 0 on error.
*/

int bufr_encode_sections0125 (sect_1_t* s1, bufr_t* msg)
{

    char** sec = msg->sec;
    int* secl = msg->secl;

    size_t st;
    int i, hand;
    long len;
    time_t t;
    struct tm t1;

    /* encode section 1. */

    hand = bitio_o_open ();
    if (hand == -1) return 0;
    if (_bufr_edition >= 4) {
         bitio_o_append (hand, 22L, 24);         /* length of section */
    }
    else {
        bitio_o_append (hand, 18L, 24);         /* length of section */
    }
    bitio_o_append (hand, s1->mtab, 8);     /* master table used */
    if (_bufr_edition >= 4) {
        bitio_o_append (hand, s1->gencent, 16);  /* originating/generating 
                                                   center */
        bitio_o_append (hand, s1->subcent, 16);  /* originating/generating
                                                   subcenter */
    }
    else {
        bitio_o_append (hand, s1->subcent, 8);  /* originating subcenter */
        bitio_o_append (hand, s1->gencent, 8);  /* originating/generating 
                                                   center */
    }
    bitio_o_append (hand, s1->updsequ, 8);  /* original BUFR message */
    bitio_o_append (hand, s1->opsec, 8);    /* no optional section */
    bitio_o_append (hand, s1->dcat, 8);     /* message type */
    if (_bufr_edition >= 4)
        bitio_o_append (hand, s1->idcatst, 8);   /* international message 
                                                    subtype */
    bitio_o_append (hand, s1->dcatst, 8);   /* local message subtype */
    bitio_o_append (hand, s1->vmtab, 8);    /* version number of master table*/
    bitio_o_append (hand, s1->vltab, 8);    /* version number of local table */

    /* if not given in section1-file take system time */

    if (s1->year == 999) {   
        time (&t);
        memcpy (&t1, localtime (&t), sizeof (struct tm));
        if (_bufr_edition >= 4) {
            bitio_o_append (hand, (long) t1.tm_year + 1900, 16); /* year */
        }
        else {
            t1.tm_year = (t1.tm_year - 1) % 100 + 1;
            bitio_o_append (hand, (long) t1.tm_year, 8);      /* year */
        }
        bitio_o_append (hand, (long) t1.tm_mon + 1, 8);       /* month */
        bitio_o_append (hand, (long) t1.tm_mday, 8);          /* day */
        bitio_o_append (hand, (long) t1.tm_hour, 8);          /* hour */
        bitio_o_append (hand, (long) t1.tm_min, 8);           /* minute */
        if (_bufr_edition >= 4) 
            bitio_o_append (hand, (long) t1.tm_sec, 8);       /* seconds */
    }
    else {
        if (_bufr_edition >= 4) {
            bitio_o_append (hand, s1->year, 16);              /* year */
        }
        else {
            s1->year = (s1->year - 1) % 100 + 1;
            bitio_o_append (hand, s1->year, 8);                /* year */
        }
        bitio_o_append (hand, s1->mon, 8);                     /* month */
        bitio_o_append (hand, s1->day, 8);                     /* day */
        bitio_o_append (hand, s1->hour, 8);                    /* hour */
        bitio_o_append (hand, s1->min, 8);                     /* minute */
        if (_bufr_edition >= 4)
            bitio_o_append (hand, s1->sec, 8);                 /* second */
    }
    if (_bufr_edition < 4)
        bitio_o_append (hand, 0L, 8);                      /* filler (0) */
    sec[1] = (char *) bitio_o_close (hand, &st);
    secl[1] = (int) st;

    /* there is no section 2 */

    sec[2] = NULL;
    secl[2] = 0;

    /* create section 5 */

    hand = bitio_o_open ();
    for (i = 0; i < 4; i ++) bitio_o_append (hand, (long) '7', 8);
    sec[5] = (char *) bitio_o_close (hand, &st);
    secl[5] = (int) st;

    /* calculate total length of BUFR-message */

    secl[0] = 8;     /* section 0 not yet setup */
    len = 0L;
    for (i = 0; i < 6; i ++) len += (long) secl[i];
  

    /* create section 0 */

    hand = bitio_o_open ();
    if (hand == -1) return 0;
    bitio_o_append (hand, (unsigned long) 'B', 8);
    bitio_o_append (hand, (unsigned long) 'U', 8);
    bitio_o_append (hand, (unsigned long) 'F', 8);
    bitio_o_append (hand, (unsigned long) 'R', 8);
    bitio_o_append (hand, len, 24);          /* length of BUFR-message */
    bitio_o_append (hand, (long) _bufr_edition, 8);  /* BUFR edition number */
    sec[0] = (char *) bitio_o_close (hand, &st);
    secl[0] = (int) st;
    return 1;
}
/*===========================================================================*/
/** \ingroup basicin
    \brief This functions saves the encoded BUFR-message to a binary file

   This function takes the encoded BUFR message and writes it to a binary file.

   \param[in] msg  The complete BUFR message
   \param[in] file The filename of the destination file

   \return 1 on success, 0 on error

   \see bufr_read_file 
*/

int bufr_write_file (bufr_t* msg, const char* file)
{

    char** sec = msg->sec; 
    int* secl = msg->secl;
    FILE *fp;
    int i;

    /* open file */

    fp = fopen (file, "wb");
    if (fp == NULL) {
        fprintf (stderr, "Could not open file %s!\n", file);
        return 0;
    }

    /* output all sections */

    for (i = 0; i < 6; i ++) {
        if (fwrite (sec[i], 1, (size_t) secl[i], fp) != (size_t) secl[i]) {
            fclose (fp);
            fprintf (stderr, 
             "An error occoured during writing '%s'. File is invalid !\n", 
                     file);
            return 0;

        }
    }

    /* close file and return */

    fclose (fp);
    return 1;
}

/*===========================================================================*/

/** \ingroup utils_g
    \brief Frees memory allocated for a BUFR message.

    This function frees all memory allocated for a BUFR message 
    by \ref bufr_data_from_file, \ref   bufr_encode_sections0125, \ref 
    bufr_read_file or \ref bufr_get_sections.

    \param[in] msg The encoded BUFR message

*/


void bufr_free_data (bufr_t* msg) {

    int i;

    if (msg == (bufr_t*) NULL) return;

    for (i = 0; i <= 5; i++) {
        if (msg->sec[i] != NULL) 
            free (msg->sec[i]);
    }
    memset (msg, 0, sizeof (bufr_t));
}



/*===========================================================================*/
/** \ingroup utils_g
    \brief Tests equality of descriptor d with (f,x,y) 

    This functions tests wheter a descriptor equals the given values f, x, y

    \param[in] d The descriptor to be tested
    \param[in] ff, xx, yy The values for testing

    \retval 1 If the descriptor equals the given values
    \retval 0 If the descriptor is different to the given values
*/

int bufr_check_fxy(dd *d, int ff, int xx, int yy) {

    if (d == (dd*) NULL) return -1;
    return (d->f == ff) && (d->x == xx) && (d->y == yy);
}



/*===========================================================================*/
/** \ingroup basicout
    \brief This function decodes sections 0 and 1.

    This function decodes sections 0 and 1 of a BUFR message.
    The BUFR edition is read from section 0 and is written to the global
    \ref _bufr_edition parameter. \n

    \param[in,out] s1 \ref sect_1_t structure to contain section 1 data
    \param[in]     msg BUFR message where the sections are stored.

    \return 1 on success, 0 on error.
*/

int bufr_decode_sections01 (sect_1_t* s1, bufr_t* msg)

{
    int h, edition;
    unsigned long l;

    /* section 0 */
    h = bitio_i_open (msg->sec[0], (size_t) msg->secl[0]);
    if (h == -1) return 0;

    bitio_i_input (h, &l, 32);                  /* BUFR */
    bitio_i_input (h, &l, 24);                  /* length of BUFR-message */
    bitio_i_input (h, &l, 8); edition = l;      /* BUFR edition number */
    bitio_i_close (h);
 
    /* section 1 */

    h = bitio_i_open (msg->sec[1], (size_t) msg->secl[1]);
    if (h == -1) return 0;
  
    bitio_i_input (h, &l, 24);                 /* length of section */

    bitio_i_input (h, &l, 8);  s1->mtab = l;    /* master table used */
    if (edition >= 4) {
        bitio_i_input (h, &l, 16);  s1->gencent = l; /* generating center */
        bitio_i_input (h, &l, 16);  s1->subcent = l; /*originating subcenter */
    }
    else {
        bitio_i_input (h, &l, 8);  s1->subcent = l; /* originating subcenter */
        bitio_i_input (h, &l, 8);  s1->gencent = l; /* generating center */
    }
    bitio_i_input (h, &l, 8);  s1->updsequ = l; /* original BUFR message */
    bitio_i_input (h, &l, 8);  s1->opsec = l;   /* no optional section */
    bitio_i_input (h, &l, 8);  s1->dcat = l;    /* message type */
    if (edition >= 4)
        bitio_i_input (h, &l, 8);  s1->idcatst = l;  /* international message 
                                                        sub type */
    bitio_i_input (h, &l, 8);  s1->dcatst = l;  /* local message subtype */
    bitio_i_input (h, &l, 8);  s1->vmtab = l;   /* version number of master 
                                                   table used */
    bitio_i_input (h, &l, 8);  s1->vltab = l;   /* version number of local 
                                                   table used */
    if (edition >= 4) {
        bitio_i_input (h, &l, 16);  s1->year = l;    /* year */
    } else {
        bitio_i_input (h, &l, 8);  s1->year = l;    /* year */
    }
    bitio_i_input (h, &l, 8);  s1->mon = l;     /* month */
    bitio_i_input (h, &l, 8);  s1->day = l;     /* day */
    bitio_i_input (h, &l, 8);  s1->hour = l;    /* hour */
    bitio_i_input (h, &l, 8);  s1->min = l;     /* minute */
    if (edition >= 4)
        bitio_i_input (h, &l, 8);  s1->sec = l;    /* second */
    bitio_i_close (h);

    /* set edition */

    _bufr_edition = edition;

    return 1;
}

/*===========================================================================*/
/** \ingroup extout
    \brief Writes section 1 data to an ASCII file

    This function writes section 1 data to an ASCII file

    \param[in]     s1     Structure where section 1 data is stored.
    \param[in]     file   Filename of the output file.

    \see bufr_sect_1_from_file
*/

int bufr_sect_1_to_file (sect_1_t* s1, const char* file) {

    FILE* fp;

    fp = fopen (file, "w");
    if (fp == NULL) {
        fprintf (stderr, "unable to open output file for section 1 !\n");
        return 0;
    }

    fprintf (fp, "%5d    master table used                  \n", s1->mtab);
    fprintf (fp, "%5d    originating subcenter              \n", s1->subcent);
    fprintf (fp, "%5d    generating center                  \n", s1->gencent);
    fprintf (fp, "%5d    original BUFR message              \n", s1->updsequ);
    fprintf (fp, "%5d    no optional section                \n", s1->opsec);
    fprintf (fp, "%5d    message type                       \n", s1->dcat);
    fprintf (fp, "%5d    local message subtype              \n", s1->dcatst);
    fprintf (fp, "%5d    version number of master table used\n", s1->vmtab);
    fprintf (fp, "%5d    version number of local table used \n", s1->vltab);
    fprintf (fp, "%5d    year                               \n", s1->year);
    fprintf (fp, "%5d    month                              \n", s1->mon);
    fprintf (fp, "%5d    day                                \n", s1->day);
    fprintf (fp, "%5d    hour                               \n", s1->hour);
    fprintf (fp, "%5d    minute                             \n", s1->min);
    /* new fields for bufr edition 4 */
    if (_bufr_edition >= 4) {
        fprintf (fp, "%5d    second                             \n", s1->sec);
        fprintf (fp, "%5d    international message subtype      \n", 
                 s1->idcatst);
    }

    fclose (fp);

    return 1;
}
 
/*===========================================================================*/
/** \ingroup basicout
    \brief Decode BUFR data and descriptor section and write values and 
    descriptors to arrays

    This function decodes the data and descriptor sections of a BUFR message
    and stored them into arrays \p descr and \p vals.
    Memory for storing descriptor- and data-array is
    allocated by this function and has to be freed by the calling function.

    \param[in]  datasec  Is where the data-section is stored.

    \param[in]  ddsec    Is where the data-descriptor-section is stored.

    \param[in]  datasecl Number of bytes of the data-section.

    \param[in]  ddescl   Number of bytes of the data-descriptor-section.

    \param[out] descr    Array where the data-descriptors are stored 
                         after reading them from the data-descriptor section. 
                         This memory area is allocated by this
                         function and has to be freed by the calling function.

    \param[out] ndescs   Number of data-descriptors in \p descs

    \param[out] vals     Array where the data corresponding to the 
                         data-descriptors is stored.

    \param[out] nvals    Number of values in \p vals

    \return
    1 if both sections were decoded successfuly, 0 on error

    \see bufr_create_msg, bufr_data_to_file

    \todo: write new version that uses bufr_t structure for output 

*/


int bufr_read_msg (void* datasec, void* ddsec, size_t datasecl, size_t ddescl,
                   dd** descr, int* ndescs, varfl** vals, size_t* nvals)


{
    int ok = 0, desch, subsets;
    dd *d;
    bufr_t msg;
    bufrval_t* bufrvals;

    memset (&msg, 0, sizeof (bufr_t));

    msg.sec[3] = ddsec;
    msg.secl[3] = (int) ddescl;
    msg.sec[4] = datasec;
    msg.secl[4] = (int) datasecl;

    /* open bitstreams for section 3 and 4 */

    desch = bufr_open_descsec_r (&msg, &subsets); 
    if (desch < 0) 
        return 0;

    if (bufr_open_datasect_r (&msg) < 0) {
        bufr_close_descsec_r (desch);
        return 0;
    }

    /* calculate number of data descriptors  */
    
    *ndescs = bufr_get_ndescs (&msg);

    /* allocate memory and read data descriptors from bitstream */

    ok = bufr_in_descsec (descr, *ndescs, desch);

    /* Input data from data-section according to the data-descriptors */

    *vals = NULL;
    *nvals = 0;
    d = *descr;

    bufrvals = bufr_open_val_array ();

    if (bufrvals == (bufrval_t*) NULL) {
        ok = 0;
    }

    if (ok) {
        while (subsets--)
        {
            ok = bufr_parse_out (d, 0, *ndescs - 1, bufr_val_to_global, 0);
            if (!ok)
                fprintf (stderr, "Error reading data from data-section !\n");
        }
        *vals = bufrvals->vals;
        *nvals = (size_t) bufrvals->nvals;
        bufrvals->vals = (varfl*) NULL;
        bufr_close_val_array ();
    }

    /* close bitstreams */

    bufr_close_descsec_r (desch);
    bufr_close_datasect_r ();

    return ok;
}

/*===========================================================================*/
/** \ingroup extout
    \brief Read descriptor section of a BUFR message from the bitsream 

    This function reads the descriptor section of a BUFR message 
    from the bitsream which was opened using \ref bufr_open_descsec_r

    \param[in,out] descs Array to hold the data descriptors
    \param[in]     ndescs Number of descriptors
    \param[in]     desch  Handle to the bitstream

    \return 1 on success, 0 on error

    \see bufr_get_ndescs, bufr_open_descsec_r, bufr_out_descsec
*/

int bufr_in_descsec (dd** descs, int ndescs, int desch) {

    int err, i;
    unsigned long l = 0;
    dd* d;

    if (desch < 0) {
        fprintf (stderr, "Descriptor handle not available! \n");
        return 0;
    }


    d = *descs = (dd *) malloc (ndescs * sizeof (dd));
    if (*descs == (dd*) NULL) {
        fprintf (stderr, "Unable to allocate memory for data descriptors !\n");
        return 0;
    }

    for (i = 0; i < ndescs; i ++) {
        err = 0;
        err = err || !bitio_i_input (desch, &l, 2);
        d->f = (unsigned char) l;
        if (!err) err = err || !bitio_i_input (desch, &l, 6);
        d->x = (unsigned char) l;
        if (!err) err = err || !bitio_i_input (desch, &l, 8);
        d->y = (unsigned char) l;
        if (err) {
            fprintf (stderr, 
                     "Number of bits for descriptor-section exceeded !\n");
            free (*descs);
            *descs = (dd*) NULL;
            return 0;
        }
        d ++;
    }
    return 1;
}
/*===========================================================================*/
/** \ingroup extout
    \brief Open bitstream of section 3 for reading 
   
    This function opens a bitstream for reading of section 3. It must be
    closed by \ref bufr_close_descsec_r.

    \param[in] msg The encoded BUFR message

    \return Returns handle to the bitstream or -1 on error

    \see bufr_close_descsec_r, bufr_in_descsec

*/

int bufr_open_descsec_r (bufr_t* msg, int *subsets) {
    
    unsigned long l;
    int desch;

    /* open bitstream */

    desch = bitio_i_open (msg->sec[3], msg->secl[3]);

    if (desch == -1) {
        bitio_i_close (desch);
        return -1;
    }

    /* skip first 7 octets (56 bits) */

    bitio_i_input (desch, &l, 24); /* length of section */
    bitio_i_input (desch, &l, 8);  /* reserved */
    bitio_i_input (desch, &l, 16); /* number of subset */
    if (subsets != NULL)
        *subsets = l;
    bitio_i_input (desch, &l, 8);  /* flags */

    return desch;
}


/*===========================================================================*/
/** \ingroup extout
    \brief close bitstream for section 3 

    This functin closes the input bitstream of section 3 which was opened by
    \ref bufr_open_descsec_r.

    \param[in] desch Handle to the bitstream

    \see bufr_open_descsec_r, bufr_in_descsec
*/

void bufr_close_descsec_r (int desch) {

    if (desch == -1) return;
    bitio_i_close (desch);
}

/*===========================================================================*/
/** \ingroup deprecated_g
    \deprecated use \ref bufr_val_to_array instead.

    This function stores the value V to an array of floats VALS. The memory-
    block for VALS is allocated in this function and has to be freed by the
    calling function.

    \param[in,out] vals The array containing the values
    \param[in]     v    The value to be put into the array
    \param[in,out] nvals Number of values in the array

    \return 1 on success, 0 on error.
*/


int val_to_array (varfl** vals, varfl v, size_t* nvals)

{
  static unsigned int nv;         /* Number of values already read from bitstream */
  static unsigned int memsize;    /* Current size of memory-block holding data-values */
  varfl *d;

/* Allocate memory if not yet done */

  if (*vals == NULL) {
    *vals = (varfl *) malloc (MEMBLOCK * sizeof (varfl));
    if (*vals == NULL) return 0;
		memset (*vals, 0, MEMBLOCK * sizeof (varfl));
    nv = 0;
    memsize = MEMBLOCK;
  }

/* Check if memory block is large anough to hold new data */

  if (memsize == nv) {
    *vals = (varfl *) realloc (*vals, (memsize + MEMBLOCK) * sizeof (varfl));
    if (*vals == NULL) return 0;
		memset ((char *) (*vals + memsize), 0, MEMBLOCK * sizeof (varfl));
    memsize += MEMBLOCK;
    if (memsize - 1 > (~(unsigned int) 0) / sizeof (varfl)) {
      fprintf (stderr, "VAL_TO_ARRAY failed in file %s, line %d\n", __FILE__, __LINE__);
      fprintf (stderr, "Try to define varfl as float in file desc.h \n");
      return 0;
    }
  }

/* Add value to array */

  d = *vals;
  *(d + nv) = v;
  nv ++;
  *nvals = nv;
  return 1;
}

/*===========================================================================*/
/** \ingroup utils_g
    \brief Store a value to an array of floats.

    This function stores the value \p v to an array of floats \p vals. 
    The memory-block for \p vals is allocated in this function and has 
    to be freed by the calling function.
    The number of values is used to calculate the size of the array
    and reallocate memory if necessary.

    \param[in,out] vals The array containing the values
    \param[in]     v    The value to be put into the array
    \param[in,out] nv   Current number of values in the array

    \return 1 on success, 0 on error.
*/


int bufr_val_to_array (varfl** vals, varfl v, int* nv)
{
    /* Allocate memory if not yet done */

    if (*vals == (varfl*) NULL) {
        *vals = (varfl *) malloc (MEMBLOCK * sizeof (varfl));
        if (*vals == (varfl*) NULL) {
            fprintf (stderr, "Could not allocate memory for value array!\n");
            return 0;
        }
		memset (*vals, 0, MEMBLOCK * sizeof (varfl));
        *nv = 0;
    }

    /* Check if memory block is large anough to hold new data */

    if (*nv != 0 && *nv % MEMBLOCK == 0) {
        *vals = (varfl*) realloc (*vals, (*nv + MEMBLOCK) * sizeof (varfl));
        if (*vals == (varfl*) NULL) {
            fprintf (stderr, "Could not allocate memory for value array!\n");
            return 0;
        }
		memset ((varfl*) (*vals + *nv), 0, MEMBLOCK * sizeof (varfl));
    }


    /* Add value to array */

    (*vals)[*nv] = v;
    (*nv)++;
    return 1;
}

/*===========================================================================*/
/** \ingroup utils_g
    \brief Store a descriptor to an array.

    This function stores the descriptor \p d to an array of descriptors
    \p descs. 
    The array descs must be large enough to hold \p ndescs + 1 descriptors.

    \param[in]     descs The array containing the descriptors
    \param[in]     d     The descriptor to be put into the array
    \param[in,out] ndescs   Current number of descriptors in the array

    \return 1 on success, 0 on error.
*/


int bufr_desc_to_array (dd* descs, dd d, int* ndescs)
{

    if (*ndescs >= MAX_DESCS) {
        fprintf (stderr, "Maximum number of descriptors exceeded!\n");
        return 0;
    }


    /* Add descriptor to array */

    descs[(*ndescs)++] = d;
    return 1;
}


/*===========================================================================*/
/** \ingroup extout
    \brief Calculate number of data descriptors in a BUFR message

    This function calculates the number of data descriptors in a BUFR
    message.

    \param[in] msg The complete BUFR message

    \return Returns the number of data descriptors.

    \see bufr_in_descsec

*/

int bufr_get_ndescs (bufr_t* msg) {

    if (msg == (bufr_t*) NULL) {
        fprintf (stderr, "Error in bufr_get_ndescs!\n");
        return -1;
    }
    return (((msg->secl[3] - 7)* 8) / 16);  
}

/*===========================================================================*/
/** \ingroup utils_g
    \brief Recall date/time info of the last BUFR-message created

   This function can be called to recall the data/time-info of the
   last BUFR-message created, if the appropiate data descriptors have
   been used.

   \param[out] year 4 digit year if \ref _bufr_edition is set to 4, 
                    year of century (2 digit) if \ref _bufr_edition is < 4.
   \param[out] mon  Month (1 - 12)
   \param[out] day  (1 - 31)
   \param[out] hour 
   \param[out] min
*/


void bufr_get_date_time (long *year, long *mon, long *day, long *hour,
                         long *min)


{
  *year = year_;
  *mon  = mon_;
  *day  = day_;
  *hour = hour_;
  *min  = min_;
}

/*===========================================================================*/
/* callback functions */
/*===========================================================================*/


/** \ingroup cbin
    \brief Outputs one data-value to the data-bitstream.

    This function outputs one data-value to the data-bitstream
    which has to be opened using \ref bufr_open_datasect_w.

    \param[in] val    Data-value to be output.
    \param[in] ind    Index to the global array 
                      \ref des[] holding the description of
                      known data-descriptors.
    
    \return 1 on success, 0 on a fault.

    \see bufr_open_datasect_w, bufr_close_datasect_w, 
         bufr_val_from_datasect
*/

static int bufr_val_to_datasect (varfl val, int ind)


{
    unsigned long l;
    int ret, wi, scale, ccitt, no_change = 0;
    varfl refval;

    assert (datah_ >= 0);

    ret = 1;
  
    /* No output for special descriptors and sequence descs*/

    if (ind == _desc_special || des[ind]->id == SEQDESC) return 1;

    /* No data width or scale change for 0 31 y, code tables, flag tables
       and ascii data */

    if (_bufr_edition < 3) {
        
        no_change = (des[ind]->el->d.f == 0 && des[ind]->el->d.x == 31);
    }
    else {
        ccitt = (strcmp (des[ind]->el->unit, "CCITT IA5") == 0 || 
                 ind == ccitt_special);
        no_change = ((des[ind]->el->d.f == 0 && des[ind]->el->d.x == 31) ||
                     ccitt || desc_is_codetable(ind) || desc_is_flagtable(ind) ||
                     ind == add_f_special || ind == cf_special);
    }

    if (no_change) {
        wi = des[ind]->el->dw;
        scale = des[ind]->el->scale;
        refval = des[ind]->el->refval;
    }
    else {
        wi = des[ind]->el->dw + dw - 128;
        scale = des[ind]->el->scale + sc - 128;
        refval = des[ind]->el->refval;
        if (incr_scale > 0)
        {
            wi = des[ind]->el->dw + (10 * incr_scale + 2) / 3;
            scale = des[ind]->el->scale + incr_scale;
            refval = des[ind]->el->refval * pow (10, incr_scale);
        }
    }

    /* If this is a missing value set all bits to 1 */
    
    if (val == MISSVAL) {
        l = 0xffffffff;
        if (bitio_o_append (datah_, l, wi) == -1) ret = 0;
    }
    /* Else it is a "normal" value */

    else {
        if (ind == cf_special) 
        {
            if (val < 0 )
                l = (unsigned long) (-val) | 1UL << (wi - 1);
            else
                l = (unsigned long) val;
        } 
        else
            l = (unsigned long) (val * pow (10.0, (varfl) scale) - refval + 0.5);

        /* + 0.5 to round to integer values */

        if (bitio_o_append (datah_, l, wi) == -1) ret = 0;

        /* check if data width was large enough to hold data to be coded */

        if (l >> wi != 0) {
            fprintf (stderr, 
  "WARNING: Tried to code the value %ld to %d bits (Datadesc.=%2d%3d%4d) !\n", 
                     l, wi, des[ind]->el->d.f, des[ind]->el->d.x, 
                     des[ind]->el->d.y);
            fprintf (stderr, "         Decoding will fail !\n");
        }
    }

    return ret;
}

/*===========================================================================*/
/** \ingroup cbinutl
    \brief Opens bitstream for section 4 writing

    This function opens the data section bitstream for writing and 
    returns its handle.

    \return Returns the handle to the data section bitstream or -1
            on error.

    \see bufr_close_datasect_w, bufr_parse_in
*/

int bufr_open_datasect_w() {
    size_t n;

    if (datah_ >= 0) {
        fprintf (stderr, "Global data handle not available.\n");
        return -1;
    }

    /* open bitstream */

    datah_ = bitio_o_open ();
    if (datah_ == -1) {
        bitio_o_close (datah_, &n);
        return -1;
    }

    /* output default data */

    bitio_o_append (datah_, 0L, 24);  /* Length of section (correct value 
                                         stored by close_datasect_w) */
    bitio_o_append (datah_, 0L, 8);   /* reserved octet, set to 0 */
    return datah_;
}

/*===========================================================================*/
/** \ingroup cboututl
    \brief Opens bitstream for reading section 4

    This function opens the data section bitstream at for reading 
    and returns its handle.

    \param[in] msg The BUFR message containing the data section.

    \return Returns the handle to the data section bitstream or -1 on error.

    \see bufr_close_datasect_r, bufr_parse_out
*/

int bufr_open_datasect_r (bufr_t* msg) {
    
    unsigned long l;
    int i;

    if (datah_ >= 0) {
        fprintf (stderr, "Global data handle not available.\n");
        return -1;
    }

    /* open bitstream */

    datah_ = bitio_i_open (msg->sec[4], (size_t) msg->secl[4]);

    if (datah_ == -1) {
        bitio_i_close (datah_);
        return -1;
    }

    /* skip trailing 4 octets (32 bits) */

    for (i = 0; i < 32; i ++) bitio_i_input (datah_, &l, 1);

    return datah_;
}
/*===========================================================================*/
/** \ingroup cbinutl
    \brief Closes bitstream for section 4 and adds data to BUFR message

    This function closes the data section bitstream
    and appends it to a BUFR message, also stores the length in the
    BUFR message.

    \param[in,out] msg BUFR message where the data has to be stored

    \see bufr_open_datasect_w, bufr_parse_in
*/

/* write length of section 4 and close bitstream */

void bufr_close_datasect_w(bufr_t* msg) {

    int n;
    size_t st;

    if (datah_ == -1 || msg == (bufr_t*) NULL) return;

    /* get current length */

    n = (int) bitio_o_get_size (datah_);

    /* number of bytes must be an even number */

	if (n % 2 != 0) bitio_o_append (datah_, 0L, 8);

    /* write length of section to beginning */
    
    n = (int) bitio_o_get_size (datah_);
    bitio_o_outp (datah_, (long) n, 24, 0L);

    /* close bitstream and return pointer */

    msg->sec[4] = (char *) bitio_o_close (datah_, &st);
    msg->secl[4] = (int) st;
    datah_ = -1;
}

/*===========================================================================*/
/** \ingroup cboututl
    \brief Closes bitstream for section 4

    This function closes the data section bitstream.

    \see bufr_open_datasect_r, bufr_parse_out
*/

void bufr_close_datasect_r () {

    if (datah_ == -1) return;
    bitio_i_close (datah_);
    datah_ = -1;
}
/*===========================================================================*/
/** \ingroup cbin
    \brief Get one value from global array of values.

    This functions gets the next value from the global array of values.

    \param[out] val The received value
    \param[in] ind    Index to the global array 
                      \ref des[] holding the description of
                      known data-descriptors.

    \return 1 on success, 0 on error.

    \see bufr_open_val_array, bufr_close_val_array
*/

int bufr_val_from_global (varfl *val, int ind) {

    assert (val != (varfl*) NULL);
    assert (vals_ != NULL);
    assert (vals_->vals != NULL);

    /* No input for special descriptors and sequence descs*/

    if (ind == _desc_special || des[ind]->id == SEQDESC) return 1;

    *val = *(vals_->vals + vals_->vali++);
    return 1;

}


/*===========================================================================*/
/** \ingroup cbout
    \brief Write one value to global array of values.

    This functions writes one value to the global array of values.

    \param[in] val    The value to store
    \param[in] ind    Index to the global array 
                      \ref des[] holding the description of
                      known data-descriptors.

    \return 1 on success, 0 on error.

    \see bufr_open_val_array, bufr_close_val_array
*/

int bufr_val_to_global (varfl val, int ind) {

    assert (vals_ != (bufrval_t*) NULL);

    /* No output for special descriptors and sequence descs*/

    if (ind == _desc_special || des[ind]->id == SEQDESC) return 1;

    return bufr_val_to_array (&(vals_->vals), val, &(vals_->nvals));
}


/*===========================================================================*/
/** \ingroup cbinutl
    \brief Opens global array of values for read/write

    This function opens the global array of values for use by 
    \ref bufr_val_from_global and \ref bufr_val_to_global and
    returns its pointer.

    \return Pointer to the array of values or NULL on error.

    \see bufr_close_val_array, bufr_val_to_global, #
    bufr_val_from_global
*/

bufrval_t* bufr_open_val_array () {


    if (vals_ != (bufrval_t*)  NULL) {
        fprintf (stderr, "Value array not empty!\n");
        return (bufrval_t*) NULL;
    }
    vals_ = malloc (sizeof (bufrval_t));

    if (vals_ == (bufrval_t*)  NULL) {
        fprintf (stderr, "Error allocating memory for Value array!\n");
        return (bufrval_t*) NULL;
    }
    memset (vals_, 0, sizeof (bufrval_t));
    return vals_;
}
/*===========================================================================*/
/** \ingroup cbinutl
    \brief Closes global array of values and frees all memory

    This function closes the global array of values used by 
    \ref bufr_val_from_global and \ref bufr_val_to_global and
    frees all allocated memory.
    
    \see bufr_open_val_array, bufr_val_to_global, bufr_val_from_global
*/

void bufr_close_val_array () {

    if (vals_ == (bufrval_t*) NULL) return;
        
    if (vals_->vals != (varfl*) NULL) {
        free (vals_->vals);
        vals_->vals = (varfl*) NULL;
    }
    free (vals_);
    vals_ = (bufrval_t*) NULL;
}

/*===========================================================================*/
/** \ingroup cbout
    \brief Reads a single value from the data stream.

    This function outputs one data-value to the data stream which was 
    opened using \ref bufr_open_datasect_r.

    \param[out] val   Data-value read.
    \param[in] ind    Index to the global array 
                      \ref des[] holding the description of
                      known data-descriptors.

    \return 1 on success, 0 on a fault.

    \see bufr_open_datasect_r, bufr_close_datasect_r, 
         bufr_val_to_datasect

*/

static int bufr_val_from_datasect (varfl *val, int ind)


{
    int data_width;
    int scale, no_change = 0, ccitt;
    unsigned long l, mv;
    varfl refval;

    assert (datah_ >= 0);

    /* No input for special descriptors and sequence descs*/

    if (ind == _desc_special || des[ind]->id == SEQDESC) return 1;

    /* No data width or scale change for 0 31 y, code tables, flag tables
       and ascii data */

    if (_bufr_edition < 3) {
        
        no_change = (des[ind]->el->d.f == 0 && des[ind]->el->d.x == 31);
    }
    else {
        ccitt = (strcmp (des[ind]->el->unit, "CCITT IA5") == 0 || 
                 ind == ccitt_special);
        no_change = ((des[ind]->el->d.f == 0 && des[ind]->el->d.x == 31) ||
                        ccitt || desc_is_codetable(ind) || desc_is_flagtable(ind) ||
                        ind == add_f_special || ind == cf_special);
    }

    if (no_change) {
        data_width = des[ind]->el->dw;
        scale = des[ind]->el->scale;
        refval = des[ind]->el->refval;
    }
    else {
        data_width = des[ind]->el->dw + dw - 128;
        scale = des[ind]->el->scale + sc - 128;
        refval = des[ind]->el->refval;
        if (incr_scale > 0)
        {
            data_width = des[ind]->el->dw + (10 * incr_scale + 2) / 3;
            scale = des[ind]->el->scale + incr_scale;
            refval = des[ind]->el->refval * pow (10, incr_scale);
        }
    }
    
    if (!bitio_i_input (datah_, &l, data_width)) {
      fprintf (stderr, "Error reading data from bitstream !\n");
      return 0;
    }
  
    /* Check for a missing value. Missval for operator qualifiers is not 
       possible */
    /* no missval for pixel values in bitmaps */
  
    mv = (1UL << data_width) - 1;

    if (l == mv && des[ind]->el->d.x != 31 && ! _opera_mode) /*
        !(des[ind]->el->d.x == 30 && des[ind]->el->d.y <= 4 && _opera_mode) &&
        !(des[ind]->el->d.x == 13 && des[ind]->el->d.y == 11 && _opera_mode) &&
        !(des[ind]->el->d.x == 21 && des[ind]->el->d.y == 14 && _opera_mode)) */ {
            *val = MISSVAL;
    }
    else if (ind == cf_special) 
    {
        *val = l & ((1UL << (data_width - 1)) - 1);
        if (l & (1UL << (data_width - 1)))
            *val = -*val;
    }
    else {
        *val = ((varfl) l + refval) / pow (10.0, (varfl) (scale));
    }
    return 1;
}

/*===========================================================================*/
/* local functions */
/*===========================================================================*/



/*===========================================================================*/
/** This function reads from a bufr-message the length of data- and
   data-descriptor-section. Therefore the buffer is opened as a bitstream
   and data is read.

   \param[in] buf   Memory-area containing the BUFR-message.
   \param[in] len   Number of bytes of the complete BUFR-message 
   determined from the length ob the input-file.
   \param[out] secl Array containing the lengths of the BUFR-sections.

   \return 1 on success, 0 on a fault.
*/

static int get_lens (char* buf, long len, int* secl)

{
    int h, co, i, totlen, lens0, ed, opt;
    unsigned long l;
    long sum;

    /* The length of section 0 is constant, but get the length of the
       whole BUFR message */

    h = bitio_i_open (buf, 8);
    bitio_i_input (h, &l, 32);        /* skip that 'BUFR' */
    bitio_i_input (h, &l, 24);        /* length of whole message */
    lens0 = l;
    bitio_i_input (h, &l, 8);         /* BUFR edition */
    ed = l;
    bitio_i_close (h);

    secl[0] = 8;
    co = 8;
    sum = 8;

    /* length of section 1 */

    h = bitio_i_open (buf + co, 20);
    if (h == -1) return 0;
    bitio_i_input (h, &l, 24);
    secl[1] = (int) l;
    co += secl[1];
    bitio_i_input (h, &l, 32);
    if (ed >= 4)
        bitio_i_input (h, &l, 16);
    bitio_i_input (h, &l, 1);
    opt = l;
    bitio_i_close (h);
    sum += secl[1];
    if (sum > len) goto err;

    /* section 2 is optional */

    secl[2] = 0;
    if (opt)
    {
        h = bitio_i_open (buf + co, 20);
        if (h == -1) return 0;
        bitio_i_input (h, &l, 24);
        secl[2] = (int) l;
        bitio_i_close (h);
        co += secl[2];
        sum += l;
        if (sum > len) goto err;
    }

    /* length of section 3 */

    h = bitio_i_open (buf + co, 20);
    if (h == -1) return 0;
    bitio_i_input (h, &l, 24);
    secl[3] = (int) l;
    co += secl[3];
    bitio_i_close (h);
    sum += l;
    if (sum > len) goto err;

    /* length of section 4 */

    h = bitio_i_open (buf + co, 20);
    if (h == -1) return 0;
    bitio_i_input (h, &l, 24);
    secl[4] = (int) l;
    co += secl[4];
    bitio_i_close (h);
    sum += l;
    if (sum > len) goto err;

    /* length of section 5 is constant */

    secl[5] = 4;
    sum += 4;
    if (sum > len) goto err;

    /* Check the total length of the message against the sum of the lengths 
       of the sections. */

    totlen = 0;
    for (i = 0; i < 6; i ++) {
#ifdef VERBOSE
        fprintf (stderr, "section %d length = %d\n", i, secl[i]);
#endif
        totlen += secl[i];
    }
    if (totlen != lens0) {
        fprintf (stderr, 
           "WARNING: Total length of message doesn't match with the lengths\n"
                     "of the individual sections !\n");
    }

    return 1;

    /* Lengths of BUFR-sections not correct */

 err:
    fprintf (stderr, "Lengths of BUFR-sections > size of input-file !\n");
    return 0;
}

/* end of file */




/* desc.c */
#define READDESC_MAIN

#define DESC_SORT
/*#define DESC_USE_INDEX*/


/*===========================================================================*/
/* internal functions                                                        */
/*===========================================================================*/

static del *decode_tabb_line (char *line);
static char *get_val (char *line, int num);
static dseq *decode_tabd_line (char *line);
static void replace_chars (char *line, char oldc, char newc);
static int key (int typ, dd* d);
static void build_keys();
static void print_desc(int i);
static void free_one_desc(int i);
static char *str_lower(char *str);
static int read_bitmap_tab (char *fn);

/*===========================================================================*/
/* internal variables                                                        */
/*===========================================================================*/

/* \brief Stucture to define the OPERA bitmap descriptors */

typedef struct bm_desc_s 
{
    int f;
    int x;
    int y;
    int dw;  
} bm_desc_t;

#define MAX_BM 100
static bm_desc_t bm_desc[MAX_BM] = {{3,21,192,1},{3,21,193,1},{3,21,194,1},
                                   {3,21,195,1},{3,21,196,1},{3,21,197,1},
                                   {3,21,200,2},{3,21,202,2}};
static int bm_size = 0;

/*===========================================================================*/

/** \ingroup desc_g
    \brief Reads bufr tables from csv-files. 

    This function reads the descriptor tables from csv-files and
    stores the descriptors in a global array \ref des. Memory for the 
    descriptors is allocated by this function and has to be freed using
    \ref free_descs.\n 
    The filenames are
    generated by this function and have the
    form bufrtab{b|d}_Y.csv or loctab{b|d}_X_Y.csv where X is a value
    calculated of the originating center and subcenter. 
    (X = \p subcent * 256 + \p gencent)
    Y is the table version.

    \param[in] dir The directory where to search for tables, if NULL
               the function uses the current directory
    \param[in] vmtab Master table version number
    \param[in] vltab Local table version number.
    \param[in] subcent Originating/generating subcenter
    \param[in] gencent Originating/generating center

    \return Returns 0 on success or -1 on errors.

    \note The local tables are optional

*/
int read_tables (char *dir, int vmtab, int vltab, int subcent, int gencent)
{
    char fn[1024];
#if defined(_WIN32)
    char *sep = "\\";
#else
    char *sep = "/";
#endif
    
    if (dir == NULL)
        dir = "";

    if (strlen(dir) == 0 || dir[strlen(dir) -1] == '/' || 
        dir[strlen(dir) -1] == '\\')
        sep = "";

    /* read master tables, the filename is bufrtab[bd]_x.csv,
       where %d stands for the version number */

    sprintf (fn, "%s%sbufrtabb_%d.csv", dir, sep, vmtab);
    if (!read_tab_b (fn)) 
    {
        fprintf (stderr, "Error: unable to read master BUFR Table B !\n");
        return -1;
    }
    
    sprintf (fn, "%s%sbufrtabd_%d.csv", dir, sep, vmtab);
    if (!read_tab_d (fn)) 
    {
        fprintf (stderr, "Error: unable to read master BUFR Table D !\n");
        return -1;
    }

    /* read local tables, the filename is localtab[bd]_x_y.csv,
       where x is the originating center and y the version number 
       Note: center is a combination of generationg center + subcenter*256,
       if no matching file is found, the subcenter is set to zero 
       TODO: change for bufr edition 4 ? */
    
    if (vltab > 0)
    {    
        sprintf (fn, "%s%slocaltabb_%d_%d.csv", dir, sep, 
                 subcent * 256 + gencent, vltab);

        if (!read_tab_b (fn))
        {
            if (subcent != 0)
            {
                sprintf (fn, "%s%slocaltabb_%d_%d.csv", dir, sep, gencent, vltab);
                if (!read_tab_b (fn))
                    fprintf (stderr, "Warning: unable to read local BUFR Table B !\n");
            }
            else
                fprintf (stderr, "Warning: unable to read local BUFR Table B !\n");
        }

        sprintf (fn, "%s%slocaltabd_%d_%d.csv", dir, sep, 
                 subcent * 256 + gencent, vltab);

        if (!read_tab_d (fn)) 
        {
            if (subcent != 0)
            {
                sprintf (fn, "%s%slocaltabd_%d_%d.csv", dir, sep, gencent, vltab);
                if (!read_tab_d (fn)) 
                    fprintf (stderr, "Warning: unable to read local BUFR Table D !\n");
            }
            else
                fprintf (stderr, "Warning: unable to read local BUFR Table D !\n");
        }
    }

    read_bitmap_tables (dir, vltab, subcent, gencent);
    
    return 0;
}

/** \ingroup desc_g
    \brief Reads list of special bitmap descriptors from csv-files. 

    This function reads a list of descriptors, which are used
    to encode compressed bitmaps or arrays of float values. 
    Each line in the file has 4 parameters (f,x,y,w), where
    f,x,y define the bufr descriptors and w the encoding method.
    The following encoding methods are defined:
    1 - 1 byte pixel value (unsigned)
    2 - 2 byte pixel value (unsigned)
    4 - 4 byte float value
    8 - 8 byte double value
    
    The filenames are generated by this function and have the
    form bmtab_X.csv or bmtab_X_Y.csv where X is a value
    calculated of the originating center and subcenter. 
    (X = \p subcent * 256 + \p gencent) and Y is the table version.

    \param[in] dir The directory where to search for tables, if NULL
               the function uses the current directory
    \param[in] vltab Local table version number.
    \param[in] subcent Originating/generating subcenter
    \param[in] gencent Originating/generating center

    \return Returns 0 on success or -1 on errors.

    \note This table is optional
*/
int read_bitmap_tables (char *dir, int vltab, int subcent, int gencent)
{
    char fn[1024];
    char *name ="bmtab";
#if defined(_WIN32)
    char *sep = "\\";
#else
    char *sep = "/";
#endif

    if (dir == NULL)
        dir = "";

    if (strlen(dir) == 0 || dir[strlen(dir) -1] == '/' || 
        dir[strlen(dir) -1] == '\\')
        sep = "";

    sprintf (fn, "%s%s%s_%d_%d.csv", dir, sep, name, subcent * 256 + gencent, vltab);
    if (read_bitmap_tab (fn) == 0) 
        return 0;
    sprintf (fn, "%s%s%s_%d.csv", dir, sep, name, subcent * 256 + gencent);
    if (read_bitmap_tab (fn) == 0) 
        return 0;
    sprintf (fn, "%s%s%s_%d_%d.csv", dir, sep, name, gencent, vltab);
    if (read_bitmap_tab (fn) == 0) 
        return 0;
    sprintf (fn, "%s%s%s_%d.csv", dir, sep, name, gencent);
    if (read_bitmap_tab (fn) == 0) 
        return 0;
    return -1;
}

/*===========================================================================*/
/* \brief reads a file with special OPERA bitmap descriptor, returns 0
 *  if OK and -1 no file is found
 */

int read_bitmap_tab (char *fn)
{
    FILE *f;
    char line[200];

    if ((f = fopen (fn, "r")) == NULL)
        return -1;

    bm_size = 0;
    while (fgets (line, 200, f) != NULL && bm_size < MAX_BM)
/*            while (! feof (f) && bm_size < 100) */
/*                if (fscanf (f, "%d %d %d %d",  */
        if (sscanf (line, "%d%*[; ]%d%*[; ]%d%*[; ]%d\n",
                    &bm_desc[bm_size].f, &bm_desc[bm_size].x, 
                    &bm_desc[bm_size].y, &bm_desc[bm_size].dw) == 4)
            bm_size++;
    fclose (f);
    return 0;
}

/*===========================================================================*/

/* \brief checks for special OPERA bitmap descriptor and returns
   the type of bitmap encoding, or zero if no bitmap descriptor */

int check_bitmap_desc (dd *d)
{
    int i;
    
    for (i = 0; i < bm_size; i++)
        if (bm_desc[i].f == d->f && bm_desc[i].x == d->x && bm_desc[i].y == d->y)
            return bm_desc[i].dw;
    return 0;
}

/*===========================================================================*/

/** \ingroup desc_g
    \brief Prints the specified descriptor or all if no descriptor specified 

    This function prints all information on the specified descriptor 
    or all descriptors if no descriptor is specified. The command line arguments
    are: [-d tabdir] [-m vmtab] [-l vltab] [-o ocenter] [-s scenter] f x y

    \param[in] argc,argv Command line arguments.
    
*/

void show_desc_args (int argc, char **argv)
{
    int f = 999, x = -1, y = -1;
    int ocent = 255, scent = 255, vmtab = 11, vltab = 4;
    char * table_dir = 0;

    while (argc > 2 && argv[1][0] == '-')
    {
        if (argv[1][1] == 'd')
            table_dir = argv[2];
        else if (argv[1][1] == 'm')
            vmtab = atoi (argv[2]);
        else if (argv[1][1] == 'l')
            vltab = atoi (argv[2]);
        else if (argv[1][1] == 'o')
            ocent = atoi (argv[2]);
        else if (argv[1][1] == 's')
            scent = atoi (argv[2]);
        argc -= 2;
        argv += 2;
    }

    if (argc > 1) f = atoi (argv[1]);
    if (argc > 2) x = atoi (argv[2]);
    if (argc > 3) y = atoi (argv[3]);
    read_tables(table_dir, vmtab, vltab, scent, ocent);
    show_desc (f, x, y);
}

/*===========================================================================*/

/** \ingroup desc_g
    \brief Prints the specified descriptor or all if f = 999 

    This function prints all information on the specified descriptor 
    or all descriptors if f = 999 

    \param[in] f,x,y The descriptor to display.
    
*/

void show_desc (int f, int x, int y)
{
    if (f == 999)
    {
        for (f = 0; f < ndes; f++)
            print_desc (f);
    }
    else if (f >= 0 && x >= 0 && y >= 0)
    {
        int i;
        dd d;
        d.f = f;
        d.x = x;
        d.y = y;
        if ((i = get_index (SEQDESC, &d)) >= 0)
            print_desc (i);
        else if ((i = get_index (ELDESC, &d)) >= 0)
            print_desc (i);
        else
            fprintf (stderr, "Descriptor %d %d %d not found !\n", f, x, y);
    }
}

/*===========================================================================*/

/* Print the descriptor at index i */

static void print_desc(int i)
{
    if (i < 0 || i >= ndes) return;

    if (des[i]->id == ELDESC)
    {
        del *d = des[i]->el;
        printf ("%d %02d %03d %2d %2d %6.2f %s  %s   [%d, %d]\n", d->d.f, d->d.x, d->d.y,
                d->scale, d->dw, d->refval, d->unit, d->elname, i, des[i]->nr);
    }
    else
    {
        int j;
        dseq *d = des[i]->seq;
        printf ("%d %02d %03d  %d %02d %03d   [%d, %d]\n", d->d.f, d->d.x, d->d.y,
                d->del[0].f, d->del[0].x, d->del[0].y, i, des[i]->nr);
        for (j = 1; j < d->nel; j++)
            printf ("          %d %02d %03d\n", d->del[j].f, d->del[j].x, d->del[j].y);
    }
}

/*===========================================================================*/

/* Compare key calculation */

static int key (int typ, dd* d)
{
    return (typ << 16) + (d->f << 14) + (d->x << 8) + d->y;
}

/* Descriptor compare function (for qsort) */ 

#ifdef DESC_SORT
static int dcmp (const void *p1, const void *p2)
{
    desc *d1 = *(desc **) p1;
    desc *d2 = *(desc **) p2;
    
    return d1->key - d2->key;
}
#endif

#ifdef DESC_USE_INDEX
/* index array for fast descriptor lookup */

    int desc_index[1<<17];
#endif

/* Create sort keys and sort the descriptor table, 
   remove duplicate entries (local table overruling) */

static void build_keys()
{
    int i, n;
    if (ndes == 0)
        return;

    for (i = 0; i < ndes; i++)
    {
        if (des[i]->id == ELDESC)
            des[i]->key = key (des[i]->id, &des[i]->el->d);
        if (des[i]->id == SEQDESC)
            des[i]->key = key (des[i]->id, &des[i]->seq->d);
    }

#ifdef DESC_SORT

    /* sort descriptors and remove duplicates */
    /* keep decsriptor with higher serial number */
    
    qsort (des, ndes, sizeof (desc *), dcmp);

    for (i = 1, n = 0; i < ndes; i++)
    {
        if (des[n]->key == des[i]->key)
        {
    	    if (des[i]->nr > des[n]->nr)
    	    {
    	        free_one_desc (n);
                des[n] = des[i];
            }
            else
                free_one_desc (i);
        }
        else
            des[++n] = des[i];
    }
    ndes = n + 1;

#endif

#ifdef DESC_USE_INDEX

    /* build index of descriptors */
    
    for (i = 0; i < (1<<17); i++) 
        desc_index[i] = -1;
    for (i = 0; i < ndes; i++)
        desc_index[des[i]->key] = i;
   
#endif
}

/*===========================================================================*/

/** \ingroup desc_g
    \brief Returns the index for the given descriptor and typ 

    This function returns the index into the global \ref des array 
    of a descriptor given by parameters \p typ and \p descr.

    \param[in] typ The type of descriptor (\ref ELDESC or \ref SEQDESC).
    \param[in] descr The descriptor.

    \return The index of the descriptor in \ref des or -1 on error.
*/

int get_index (int typ, dd* descr)
{
#ifdef DESC_USE_INDEX

    int k = key (typ, descr);
    return desc_index[k];
   
#else

#ifdef DESC_SORT

    int i1 = 0;
    int i2 = ndes -1;
    int k = key (typ, descr);

    while (i2 >= i1)
    {
        int i = (i2 + i1) / 2;
	    int diff = des[i]->key - k;
        if (diff == 0)
            return i;
        if (diff < 0)
            i1 = i + 1;
        else
            i2 = i - 1;
    }
    return -1;

#else

  int i;
  int k = key (typ, descr);
  for (i = 0; i < ndes; i ++) 
  {
      if (des[i]->key == k)
        return i;
  }
  return -1;

#endif
#endif
}

/*===========================================================================*/
/** \ingroup desc_g
    \brief Reads bufr table d from a csv-files. 

    This function reads a sequence descriptor table (d) from  a csv-file and
    stores the descriptors in a global array \ref des. Memory for the 
    descriptors is allocated by this function and has to be freed using
    \ref free_descs.

    \param[in] fname The name of a csv-file.

    \return Returns 1 on success or 0 on error.

    \see read_tables, read_tab_b
*/

int read_tab_d (char *fname)

{
    FILE *fp;
    char line[1000], *l;
    dseq *sdesc;
    int end;

    /* Open input file */

    fp = fopen (fname, "r");
    if (fp == NULL) {
        fprintf (stderr, "unable to open '%s'\n", fname);
        return 0;
    }

/* Run through all lines and decode the ones that contain reasonable data */

    end = 0;
    do {
        if ((l = fgets (line, 1000, fp)) != NULL)
        {
            /* For some reasons the '-' is not correct stored in the csv file */
            replace_chars (l, -105, 45); 
            replace_chars (l, -106, 45);
        }

        sdesc = decode_tabd_line (l);
        if (sdesc != NULL) {
            des[ndes] = malloc (sizeof (desc));
            if (des[ndes] == NULL) {
                fprintf (stderr, "Memory allocation error.\n");
                fclose (fp);
                return 0;
            }
            des[ndes]->id = SEQDESC;
            des[ndes]->nr = ndes;
            des[ndes]->seq = sdesc;
            des[ndes]->el = NULL;
            ndes ++;
            if (ndes >= MAXDESC) {
                fprintf (stderr, "Parameter MAXDESC exceeded.\n");
                fclose (fp);
                return 0;
            }
        }
    } while (l != NULL);

    fclose (fp);

    build_keys();
    return 1;
}

/*===========================================================================*/
static dseq *decode_tabd_line (char *line)

/* Decodes a single Table D Line and returns a pointer to a dseq-structure
         holding the data that has been decoded. The memory area must be
         freed by the calling function
*/

{
/* Get the first 6 strings of the line, each of them separated by a ';'
*/

    char *sf, *sx, *sy, *dx, *dy, *df;
    int isf, isx, isy, idx, idy, idf;
    static dseq *seq = NULL;           /* Holds the current Sequence Descriptor */
    dseq *ret = NULL;
    dd *ddp;
    char tmp[1000];

    if (line == NULL)
    {
        ret = seq;
        seq = NULL;
        return ret;
    }

    strcpy (tmp, line);

    dy = get_val (line, 5);
    dx = get_val (line, 4);
    df = get_val (line, 3);
    sy = get_val (line, 2);
    sx = get_val (line, 1);
    sf = get_val (line, 0);

/* CHeck for valid values */

    if (dy == NULL ||
        dx == NULL ||
        df == NULL ||
        sy == NULL ||
        sx == NULL ||
        sf == NULL) return NULL;

    if (sscanf (sf, "%d", &isf) != 1) isf = 0;
    if (sscanf (sx, "%d", &isx) != 1) isx = 0;
    if (sscanf (sy, "%d", &isy) != 1) isy = 0;
    if (sscanf (df, "%d", &idf) != 1) idf = 0;
    if (sscanf (dx, "%d", &idx) != 1) idx = 0;
    if (sscanf (dy, "%d", &idy) != 1) idy = 0;

/* Check if there is a new seqence descriptor */

    if (isf == 3 || isx != 0 || isy != 0) {
        if (seq != NULL) {
            ret = seq;       /* This is what we return */
        }
        seq = malloc (sizeof (dseq));
        if (seq == NULL) {
            fprintf (stderr, "Memory allocation error !\n");
            return NULL;
        }
        seq->d.f = isf;
        seq->d.x = isx;
        seq->d.y = isy;
        seq->nel = 0;
        seq->del = malloc (sizeof (dd));
        if (seq->del == NULL) {
            fprintf (stderr, "Memory allocation error !\n");
            return NULL;
        }
    }

/* Get the new entry for the sequence */

    if ((idf != 0 || idx != 0 || idy != 0) && seq != NULL) {
        seq->del = realloc (seq->del, (seq->nel + 1) * sizeof (dd));
        if (seq->del == NULL) {
            fprintf (stderr, "Memory allocation error !\n");
            return NULL;
        }
        ddp = seq->del + seq->nel;
        ddp->f = idf;
        ddp->x = idx;
        ddp->y = idy;
        seq->nel += 1;
    }

    return ret;
}

/*===========================================================================*/
/** \ingroup desc_g
    \brief Reads bufr table b from a csv-files. 

    This function reads an element descriptor table (b) from a csv-file and
    stores the descriptors in a global array \ref des. Memory for the 
    descriptors is allocated by this function and has to be freed using
    \ref free_descs.

    \param[in] fname The name of the csv-file.

    \return Returns 1 on success or 0 on error.

    \see read_tables, read_tab_d
*/

int read_tab_b (char *fname)

{
    FILE *fp;
    char line[1000];
    del *descr;

    /* Open input file */

    fp = fopen (fname, "r");
    if (fp == NULL) {
        fprintf (stderr, "unable to open '%s'\n", fname);
        return 0;
    }

    /* Run through all lines and decode the ones that contain reasonable data*/

    while (fgets (line, 1000, fp) != NULL) {
        replace_chars (line, -106, 45); /* For some reasons the '-' is not correct stored in the csv file */
        replace_chars (line, -105, 45); /* For some reasons the '-' is not correct stored in the csv file */
        descr = decode_tabb_line (line);
        if (descr != NULL) {
            des[ndes] = malloc (sizeof (desc));
            if (des[ndes] == NULL) {
                fprintf (stderr, "Memory allocation error.\n");
                fclose (fp);
                return 0;
            }
            des[ndes]->id = ELDESC;
            des[ndes]->nr = ndes;
            des[ndes]->el = descr;
            des[ndes]->seq = NULL;
            ndes ++;
            if (ndes >= MAXDESC) {
                fprintf (stderr, "Parameter MAXDESC exceeded.\n");
                fclose (fp);
                return 0;
            }
        }
    }

    fclose (fp);

    /* Finally we add a dummy descriptor describing a single character 
       in a CCITT IA5 character string */

    if (ccitt_special == 0) {
        ccitt_special = MAXDESC + 1;
        descr = decode_tabb_line ("9999;9999;9999;tmp;value;0;0;8;tmp;0;3");
        if (descr != NULL) {
            des[ccitt_special] = malloc (sizeof (desc));
            if (des[ccitt_special] == NULL) {
                fprintf (stderr, "Memory allocation error.\n");
                return 0;
            }
            des[ccitt_special]->id = ELDESC;
            des[ccitt_special]->nr = ccitt_special;
            des[ccitt_special]->el = descr;
            des[ccitt_special]->seq = NULL;
        }
    }

    /* The same we need for a dummy for saving a change in the 
       reference value */

    if (cf_special == 0) {
        cf_special = MAXDESC + 2;
        descr = decode_tabb_line ("9999;9999;9998;Reference value;value;0;0;8;tmp;0;3");
        if (descr != NULL) {
            des[cf_special] = malloc (sizeof (desc));
            if (des[cf_special] == NULL) {
                fprintf (stderr, "Memory allocation error.\n");
                return 0;
            }
            des[cf_special]->id = ELDESC;
            des[cf_special]->nr = cf_special;
            des[cf_special]->el = descr;
            des[cf_special]->seq = NULL;
        }
    }

    /* dummy descriptor for the associated field */

    if (add_f_special == 0) {
        add_f_special = MAXDESC + 3;
        descr = decode_tabb_line ("0;0;0;Associated Field;value;0;0;0;tmp;0;0");
        if (descr != NULL) {
            des[add_f_special] = malloc (sizeof (desc));
            if (des[add_f_special] == NULL) {
                fprintf (stderr, "Memory allocation error.\n");
                return 0;
            }
            des[add_f_special]->id = ELDESC;
            des[add_f_special]->nr = add_f_special;
            des[add_f_special]->el = descr;
            des[add_f_special]->seq = NULL;
        }
    }

    /* dummy descriptor for no data output */
    
    if (_desc_special == 0) {
        _desc_special = MAXDESC + OPTDESC - 1;
        descr = decode_tabb_line ("0;0;0;Desc;value;0;0;0;tmp;0;0");
        if (descr != NULL) {
            des[_desc_special] = malloc (sizeof (desc));
            if (des[_desc_special] == NULL) {
                fprintf (stderr, "Memory allocation error.\n");
                return 0;
            }
            des[_desc_special]->id = ELDESC;
            des[_desc_special]->nr = _desc_special;
            des[_desc_special]->el = descr;
            des[_desc_special]->seq = NULL;
        }
    }

    build_keys();
    return 1;
}

/*===========================================================================*/
/** \ingroup desc_g
    \brief Frees all memory that has been allocated for data descriptors

    This function frees all memory that has been allocated for data descriptors

    \see read_tables, read_tab_b, read_tab_d
*/

void free_descs (void)


{
    int i;

    for (i = 0; i < ndes; i ++) {
        free_one_desc (i);
    }
    ndes = 0;

    free_one_desc (ccitt_special);
    free_one_desc (cf_special);
    free_one_desc (add_f_special);
    free_one_desc (_desc_special);
    ccitt_special = 0;
    cf_special = 0;
    add_f_special = 0;
    _desc_special = 0;
}

static void free_one_desc (int i)
{
    if (i < 0 || i >= MAXDESC + OPTDESC|| des[i] == NULL) 
       return;

    if (des[i]->id == ELDESC) {
        free (des[i]->el->unit);
        free (des[i]->el->elname);
        free (des[i]->el);
    }
    else if (des[i]->id == SEQDESC) {
        free (des[i]->seq->del);
        free (des[i]->seq);
    }
    free (des[i]);
    des[i] = NULL;
}

/*===========================================================================*/
static del *decode_tabb_line (char *line)

/* Decodes a single Table B Line and returns a pointer to a del-structure
         holding the data that has been decoded. The memory area must be
         freed by the calling function
*/

{
/* Get the first 8 strings of the line, each of them separated by a ';'
*/

    char *data_width, *refval, *scale, *unit, *name, *x, *y, *f;
    del desc, *ret;
    float tmp;
    char tmpline[1000];

    memset (&desc, 0, sizeof (del));
    strcpy (tmpline, line);
 
    data_width = get_val (tmpline, 7);
    refval     = get_val (tmpline, 6);
    scale      = get_val (tmpline, 5);
    unit       = get_val (tmpline, 4);
    name       = get_val (tmpline, 3);
    y          = get_val (tmpline, 2);
    x          = get_val (tmpline, 1);
    f          = get_val (tmpline, 0);

    if (data_width == NULL ||
        refval     == NULL ||
        scale      == NULL ||
        unit       == NULL ||
        name       == NULL ||
        x          == NULL ||
        y          == NULL ||
        f          == NULL) return NULL;


/* A correct line has been found decode data from strings */

    if (sscanf (f,          "%d", &desc.d.f)   != 1) return NULL;
    if (sscanf (x,          "%d", &desc.d.x)   != 1) return NULL;
    if (sscanf (y,          "%d", &desc.d.y)   != 1) return NULL;
    if (sscanf (scale,      "%d", &desc.scale) != 1) return NULL;
    if (sscanf (data_width, "%d", &desc.dw)    != 1) return NULL;
    if (sscanf (refval,     "%f", &tmp)        != 1) return NULL;
    desc.refval = tmp;

    desc.unit = malloc (strlen (unit) + 1);
    if (desc.unit == NULL) {
        fprintf (stderr, "Memory allocation error !\n");
        return NULL;
    }
    strcpy (desc.unit, unit);

    desc.elname = malloc (strlen (name) + 1);
    if (desc.elname == NULL) {
        fprintf (stderr, "Memory allocation error !\n");
        return NULL;
    }
    strcpy (desc.elname, name);

    ret = malloc (sizeof (del));
    if (ret == NULL) {
        fprintf (stderr, "Memory allocation error !\n");
        return NULL;
    }

    memcpy (ret, &desc, sizeof (del));
    return ret;
}
/*===========================================================================*/
/** Checks if a descriptor is a flag-table.
    
    \param[in] ind Index to the global array \ref des[] holding the 
                   description of known data-descriptors.

    \return 1 if descriptor is a flag-table, 0 if not.

    \see desc_is_codetable
*/

int desc_is_flagtable (int ind) {

    char unit[20];

    strncpy (unit, des[ind]->el->unit, 20);
    unit[19] = '\0';

    str_lower (unit);

    return (strcmp (unit, "flag table") == 0 ||
            strcmp (unit, "flag-table") == 0);
}

/*===========================================================================*/
/** Checks if a descriptor is a code-table.
    
    \param[in] ind Index to the global array \ref des[] holding the 
                   description of known data-descriptors.

    \return 1 if descriptor is a code-table, 0 if not.

    \see desc_is_flagtable
*/

int desc_is_codetable (int ind) {

    char unit[20];
    
    strncpy (unit, des[ind]->el->unit, 20);
    unit[19] = '\0';

    str_lower (unit);

    return (strcmp (unit, "code table") == 0 ||
            strcmp (unit, "code-table") == 0);
}

/*===========================================================================*/
static char *get_val (char *line, int num)

/* Gets a single value (character string) from a LINE licated at position
         NUM
*/

{
    char *p;
    int i;

/* seek to the end of the desired value and set it to 0 to identify the
         end of the string. */

    p = line;
    for (i = 0; i < num + 1 && p != NULL; i ++) {
        if (i == 0) p = strchr (p, ';');
        else p = strchr (p + 1, ';');
    }
    if (p != NULL) *p = 0;

/* Now seek to the beginning of the desired value */

    p = line;
    for (i = 0; i < num && p != NULL; i ++) {
        if (i == 0) p = strchr (p, ';');
        else p = strchr (p + 1, ';');
    }
    if (p == NULL) return NULL;
    if (num != 0) p ++;
    return p;
}

/*===========================================================================*/
/** \ingroup utils
    \brief Deletes all terminating blanks in a string.

    This functions deletes all terminating blanks in a string.

    \param[in,out] buf Our string.
*/

void trim (char *buf)

{
  int i, len;

  len = strlen (buf);
  for (i = len - 1; i >= 0; i --) {
    if (*(buf + i) == ' ') *(buf + i) = 0;
    else break;
  }
}

/*===========================================================================*/
/** \ingroup desc_g
    \brief Returns the unit for a given data descriptor

    This function searches the global \ref des array and returns 
    the unit for a data descriptor.

    \param[in] d The descriptor.
    
    \return Pointer to a string containing the unit or NULL if the
            descriptor is not found in the global \ref des array.
*/

char *get_unit (dd* d)

{
  int i;

  for (i = 0; i < ndes; i ++) {
    if (des[i]->id == ELDESC &&
        memcmp (d, &(des[i]->el->d), sizeof (dd)) == 0)
        return des[i]->el->unit;
  }
  return NULL;
}

/*===========================================================================*/
static void replace_chars (char *line, char oldc, char newc)

/* replaces one character of a string against another.
*/

{
    for (;*line != 0; line ++) {
         if (*line == oldc) 
             *line = newc;
    }
}

/*===========================================================================*/
/**
   Converts a given string to lower case characters.
    
   \param[in,out] *str:         pointer to the string
   \return The pointer to the start of the string
*/

static char *str_lower(char *str)
{
    register char *p = str;
    while (*p != '\0') {
        *p = (char) tolower((int) *p);
        p++;
    }
    return str;
}

/*===========================================================================*/

/* end of file */



/* Bufr_io.c*/

#define BUFR_OUT_BIN 0 /**< \brief Output to binary format for flag tables */

/* \brief Stucture that holds a decoded (source)
                                bufr message */
typedef struct bufrsrc_s {  

    sect_1_t s1;             /**< \brief section 1 information */
    dd* descs;               /**< \brief array of data descriptors */
    int ndesc;               /**< \brief number of data descriptors */
    int desci;               /**< \brief current index into descs */
    bd_t* data;              /**< \brief array of data elements */
    int ndat;                /**< \brief number of data elements */
    int datai;               /**< \brief current index into data */
} bufrsrc_t;

extern int errno;

static bufrsrc_t* src_ = NULL;   /* structure containing data for encoding */
static int nrows_ = -1;          /* number of rows for bitmap */
static int ncols_ = -1;          /* number of colums for bitmap */
static FILE *fo_ = NULL;         /* File-Pointer for outputfile */
static char* imgfile_ = NULL;    /* filename for uncompressed bitmap */
static char* char_ = NULL;       /* character array for ascii input */
static int   cc_ = 0;            /* index into char array for ascii input */

static int bufr_read_src_file (FILE* fp, bufrsrc_t* data);
#if BUFR_OUT_BIN
static void place_bin_values (varfl val, int ind, char* buf);
#endif
static int bufr_input_char (varfl* val, int ind);
static int desc_to_array (dd* d, bufrsrc_t* data);
static int string_to_array (char* s, bufrsrc_t* bufr);
static int bufr_src_in (varfl* val, int ind);
static int bufr_file_out (varfl val, int ind);
static int bufr_char_to_file (varfl val, int ind);
static bufrsrc_t* bufr_open_src ();
static void bufr_close_src ();
static FILE* bufr_open_output_file (char* name);
static void bufr_close_output_file ();
static char *str_save(char *str);
static void replace_bin_values (char *buf);

static int z_decompress_to_file (char* outfile, varfl* vals, int* nvals);
static int z_compress_from_file (char* infile, varfl* *vals, int* nvals);
static void byteswap64 (unsigned char *buf, int n);

#define MAX_LINE 2000        /* Max. linelength in input file */
#define MAX_DATA  1000000    /* Maximum number of data elements */

/*===========================================================================*/

/** \ingroup operaio
    \brief read data and descriptors from ASCII file and code them into
    sections 3 and 4

    This function reads descriptors and data from an
    ASCII file and codes them into a BUFR data descriptor and data
    section (section 3 and 4).
    Memory for both sections is allocated in 
    this function and must be freed by the calling functions using
    \ref bufr_free_data.

    \param[in] file  Name of the input ASCII file
    \param[in,out] msg  BUFR message to contain the coded sections

    \return 1 on succes, 0 on error

    \see bufr_data_to_file, bufr_create_msg, bufr_free_data

*/

int bufr_data_from_file(char* file, bufr_t* msg)
{
    FILE* fp;
    bufrsrc_t* src;
    int ok = 0, desch = -1;

    /* open file */

    fp = fopen (file, "r");
    if (fp == (FILE*) NULL) {
        fprintf (stderr, "Could not open file %s\n", file);
        return 0;
    }

    /* open global src structure for holding data from file */

    src = bufr_open_src ();
    ok = (src != (bufrsrc_t*) NULL);

    /* read data from file to arrays */

    if (ok) {
        ok = bufr_read_src_file (fp, src);
        fclose (fp);
    }

    /* output descriptors to section 3 */

    if (ok) {
        /* open bitstream */

        desch = bufr_open_descsec_w (1);        
        ok = (desch >= 0);
    }

    if (ok)
        /* write descriptors to bitstream */
        
        ok = bufr_out_descsec (src->descs, src->ndesc, desch);

         /* close bitstream and write data to msg */
    
    bufr_close_descsec_w (msg, desch);

    /* parse descriptors and encode data to section 4 */

    if (ok)
        /* open bitstream */
        
        ok = (bufr_open_datasect_w () >= 0);

    if (ok) 
        /* write data to bitstream */

        ok = bufr_parse_in (src->descs, 0, src->ndesc - 1, bufr_src_in, 1);

        /* close bitstream and write data to msg */

    bufr_close_datasect_w (msg);

    /* free src and cleanup globals */

    bufr_close_src ();

    return ok;
}

/*===========================================================================*/
/** \ingroup operaio
    \brief Decode data and descriptor sections of a BUFR message
    and write them to an ASCII file

    This functions decodes data and descriptor sections of a BUFR message
    and writes them into an ASCII file. 
    If there is an OPERA bitmap (currently descriptors 3 21 192 to 3 21 197,
    3 21 200 and 3 21 202) it is written to a seperate file.

    \param[in] file  Name of the output ASCII file
    \param[in] imgfile  Name of the output bitmap file(s)
    \param[in] msg  BUFR message to contain the coded sections

    \return 1 on succes, 0 on error

    \see bufr_data_from_file, bufr_read_msg

*/

int bufr_data_to_file (char* file, char* imgfile, bufr_t* msg) {

    dd *dds = NULL;
    int ok;
    int ndescs, desch, subsets;

    /* open output-file */

    if (bufr_open_output_file (file) == (FILE*) NULL) {
        fprintf (stderr, "Unable to open outputfile '%s'\n", file);
        return 0;
    }

    /* set image file name */

    imgfile_ = imgfile;

    /* open bitstreams for section 3 and 4 */

    desch = bufr_open_descsec_r(msg, &subsets);

    ok = (desch >= 0);

    if (ok)
        ok = (bufr_open_datasect_r(msg) >= 0);

    /* calculate number of data descriptors  */
    
    ndescs = bufr_get_ndescs (msg);

    /* allocate memory and read data descriptors from bitstream */

    if (ok) 
        ok = bufr_in_descsec (&dds, ndescs, desch);


    /* output data and descriptors */

    if (ok)
      while (subsets--) 
        ok = bufr_parse_out (dds, 0, ndescs - 1, bufr_file_out, 1);
        
    /* close bitstreams and free descriptor array */

    if (dds != (dd*) NULL)
        free (dds);
    bufr_close_descsec_r (desch);
    bufr_close_datasect_r ();
    bufr_close_output_file ();

    return ok;
}


/*===========================================================================*/
/* local functions */
/*===========================================================================*/

/** read data and descriptors from an OPERA BUFR file and store it into
   a bufrsrc_t structure 

   \param[in] fp Pointer to the source file
   \param[in,out] data Is where the data and descriptors are to be stored

   \return 1 on success, 0 on error

*/
static int bufr_read_src_file (FILE* fp, bufrsrc_t* data) {

    dd d;                 /* current descriptor */
    char buf[MAX_LINE];   /* strung containing current line */
    char s[MAX_LINE];     /* string containing data element or filename */
    char* sbuf = NULL;  
    int l, n, ascii_flag = 0;

    if (fp == NULL || data == (bufrsrc_t*) NULL) return 0;

    /* read each line and process it */

    while (fgets (buf, MAX_LINE, fp) != NULL) {

        ascii_flag = 0;
        l = strlen (buf);

        /* delete terminating \n and blanks */

        if (buf[l-1] == '\n') buf[l-1] = 0;
        trim (buf);                        
      
        /* ignore comments */

        if (buf[0] == '#' || strlen (buf) == 0)
            continue;


        /* check for ascii and binary data */

        if (strstr (buf, " '") != NULL) {

            sbuf = strstr (buf, " '");
            if (sbuf == NULL) {
                fprintf (stderr, "Error reading ASCII data from input file\n");
                return 0;
            }
            sbuf++;
            ascii_flag = 1;
        } else {
            
            /* replace binary values by integers */

            replace_bin_values (buf);
        }

        /* check for data descriptor and data */

        n = sscanf (buf, "%d %d %d %s", &d.f, &d.x, &d.y, s);


        /* replication and modification descriptors don't have values */

        if (d.f == 1 || (d.f == 2 && d.x != 5))
            n = sscanf (buf, "%d %d %d", &d.f, &d.x, &d.y);

        /* descriptor and data */

        if (n == 4) {
            if (!desc_to_array (&d, data)) return 0;
            if (ascii_flag) {
                if (!string_to_array (sbuf, data)) return 0;
            } else {
                if (!string_to_array (s, data)) return 0;
            }
        }
        /* only descriptor */

        else if (n == 3) {
            if (!desc_to_array (&d, data)) return 0;
        }
        /* only data */

        else {
            if (ascii_flag) {
                if (!string_to_array (sbuf, data)) return 0;
            } else {
                if (!sscanf (buf, "%s", s)) return 0;
                if (!string_to_array (s, data)) return 0;
            }
        }
    }
    return 1;
}

/*===========================================================================*/
/* \ingroup cbinutl
    \brief Opens bufrsrc structure for function \ref bufr_src_in

    This functions opens a structure to hold BUFR data descriptors and
    data elements from an ASCII file for use by \ref bufr_src_in
    and returns the pointer to this structure.

    \return Pointer to the BUFR src structure or NULL if an error occured.

    \see bufr_close_src, bufr_read_src, bufr_src_in
*/

    
   
static bufrsrc_t* bufr_open_src () {

    if (src_ != (bufrsrc_t*) NULL) {

        fprintf (stderr, "Global src structure not available!\n");
        return (bufrsrc_t*) NULL;
    }

    src_ = malloc (sizeof (bufrsrc_t));
    if (src_ == (bufrsrc_t*) NULL) {

        fprintf (stderr, "Error allocating memory for src structure!\n");
        return (bufrsrc_t*) NULL;
    }

    memset (src_, 0, sizeof (bufrsrc_t));

    return src_;
}

/*===========================================================================*/
/* \ingroup cbinutl
    \brief Closes bufrsrc structure for function \ref bufr_src_in

    This functions closes the structure used by \ref bufr_src_in

    \see bufr_open_src, bufr_src_in
*/

static void bufr_close_src () {

    int i;

    if (src_ == (bufrsrc_t*) NULL) return;

    if (src_->data != (bd_t*) NULL) {
        for (i = 0; i < src_->ndat; i++)
            free (src_->data[i]);
        free (src_->data);
    }

    if (src_->descs != (dd*) NULL)
        free (src_->descs);

    free (src_);
    src_ = (bufrsrc_t*) NULL;

}


/*===========================================================================*/
/* \ingroup cbin

   \brief Gets next data value from BUFR source data.

   This function

   \param[out] val The received value
   \param[in] ind   Index to the global array 
                      \ref des[] holding the description of
                      known data-descriptors.

   \return 1 on success, 0 on error
*/

static int bufr_src_in (varfl* val, int ind) {

    char* line;        /* current string we have to convert */
    int datai;         /* index to current data element */
    dd* d;             /* current descriptor */
    int depth = 0;     /* image depth in bytes per pixel for bitmaps */
    int ok = 0; 
    bufrval_t* vals;

    assert (val != (varfl*) NULL);
    assert (src_ != (bufrsrc_t*) NULL);

    /* get next line frome array */

    datai = src_->datai;
    line = src_->data[datai];
    if (line == NULL) {
        fprintf (stderr, "Data element empty!\n");
        return 0;
    }

    /* element descriptor */

    if (des[ind]->id == ELDESC) {

        d = &(des[ind]->el->d);

        /* special treatment for ASCII data */

        if (ind == _desc_special) {
            char* unit;

            unit = des[ind]->el->unit;
            if (unit != NULL && strcmp (unit, "CCITT IA5") == 0) {
                char_ = line;
                cc_ = 0;
                if (!bufr_parse_in (d, 0, 0, bufr_input_char, 0)) {
                    return 0;
                }
                /* check if we reached end of string */

                if (char_[cc_+1] != '\'') {
                    fprintf (stderr, 
                             "Number of bits missmatch for ascii data!\n");
                    return 0;
                }
                cc_ = 0;
                char_ = NULL;
                src_->datai++;
                return 1;
            }
            else {
                return 1;
            }
        }

        /* "normal" data -> get one value */

        else {

            /* check for missing */

            if (strstr (line, "missing") != NULL ||
                strstr (line, "MISSING") != NULL) {
                *val = MISSVAL;
                src_->datai++;
                return 1;
            }

            /* convert to varfl */

            errno = 0;
            *val = (varfl) strtod (line, NULL);
            src_->datai++;
            if (errno) {
                fprintf (stderr, "Error reading value from bufr_src\n");
                return 0;
            }
            
            /* check for number of rows / columns or
               bins / rays of radar bitmap */

            if (bufr_check_fxy(d, 0, 30, 21) > 0 ||
                bufr_check_fxy(d, 0, 30, 195) > 0)
                ncols_ = (int) *val;
            if (bufr_check_fxy(d, 0, 30, 22) > 0 ||
                bufr_check_fxy(d, 0, 30, 194) > 0)
                nrows_ = (int) *val;

            return 1;
        }
    }
    /* sequence descriptor */

    else if (des[ind]->id == SEQDESC) {

        /* check if bitmap or "normal" sequence descriptor */
        
        d = &(des[ind]->seq->d);
        
        depth = check_bitmap_desc(d);

        /* seqdesc is a special opera run length encoded bitmap */

        if (depth > 0) {

            if (nrows_ <= 0 || ncols_ <= 0) {
                fprintf (stderr, "Unknown number of rows and/or columns\n");
                return 0;
            }

            /* read bitmap and run length encode to memory */

            /* initialize array */

            vals = bufr_open_val_array ();

            if (vals == (bufrval_t*) NULL) return 0;

            if (depth == 8)
            {
                if (!z_compress_from_file (line, &(vals->vals), &(vals->nvals)))
                { 
                    fprintf (stderr, "Error during z-compression.\n");
                    bufr_close_val_array ();
                    return 0;
                }
            } else {
                if (!rlenc_from_file (line, nrows_, ncols_, &(vals->vals), 
                                      &(vals->nvals), depth)) 
                { 
                    fprintf (stderr, "Error during runlength-compression.\n");
                    bufr_close_val_array ();
                    return 0;
                  }
            }
            src_->datai++;

            ok = bufr_parse_in (des[ind]->seq->del, 0, des[ind]->seq->nel - 1,
                                bufr_val_from_global, 0);

            /* free array */

            bufr_close_val_array ();
            return ok;
        } 
        /* normal sequence descriptor - just call bufr_parse_in with 
           all descriptors in sequence */
        
        else {
            return bufr_parse_in (des[ind]->seq->del, 0, 
                                  des[ind]->seq->nel - 1, bufr_src_in, 1);
        }

    }
    else {
        fprintf (stderr, "Unknown descriptor in bufr_src_in!\n");
        return 0;
    }
}



/*===========================================================================*/
/** \ingroup cboututl
    \brief Opens file for ouput of BUFR data in ASCII format

    This functions opens a file for output to ASCII by \ref bufr_file_out 
    and returns its pointer.

    \param[in] name The name of the output file.
    
    \return Pointer to the file or NULL on error.

    \see bufr_file_out, bufr_close_output_file
*/

static FILE* bufr_open_output_file (char* name) {

     if (fo_ != (FILE*) NULL) {
         fprintf (stderr, "Global output file not available!\n");
         return (FILE*) NULL;
     }

     fo_ = fopen (name, "w");
     return fo_;
 }

/*===========================================================================*/
/** \ingroup cboututl
    \brief Closes ASCII file opened by \ref bufr_open_output_file

    This functions closes the ASCII output file used by \ref bufr_file_out 

    \see bufr_file_out, bufr_open_output_file
*/

static void bufr_close_output_file () {

    if (fo_ == (FILE*) NULL) return;
    fclose (fo_);
    fo_ = (FILE*) NULL;
}


/*===========================================================================*/
/** \ingroup cbout
    \brief Outputs one value + descriptor to an ASCII-file

    This function outputs data values and descriptors to an ASCII file
    opened by \ref bufr_open_output_file.
    In case of CCITT (ASCII) data it calls \ref bufr_parse_out 
    with the callback  \ref bufr_char_to_file for output of the 
    single characters. \n
    In case of sequence descriptors it checks if the descriptor is a special
    OPERA bitmap (currently descriptors 3 21 192 to 3 21 197, 3 21 200 
    and 3 21 202) and in this case writes the data to a special file 
    . For normal sequence descriptors it just
    calls  bufr_parse_out again. \n
    The function also makes use of the global \ref _replicating flag in order
    to decide whether it has to print out the data descriptors or not.

    \param[in] val    Data-value to be output.
    \param[in] ind    Index to the global array \ref des[] holding the 
                      description of known data-descriptors or special 
                      descriptor (\ref ccitt_special, _desc_special,
                      add_f_special).

    \return The function returns 1 on success, 0 on a fault.

    \see bufr_src_in, bufr_open_output_file, bufr_close_output_file,
    bufr_parse_out, _replicating
*/

static int bufr_file_out (varfl val, int ind)

{
    int depth = 1, nv, ok;
    char sval[80];
    char fname[512], tmp[80];
    char* unit;
    dd* d;
    static int nchars = 0;    /* number of characters for ccitt output */
    static int in_seq = 0;    /* flag to indicate sequences */
    static int first_in_seq;  /* flag to indicate first element in sequence */
    static int count = 0;     /* counter for image files */
    bufrval_t* vals;

    /* sanity checks */

    if (des[ind] == (desc*) NULL || fo_ == (FILE*) NULL 
        || imgfile_ == (char* ) NULL) {
        fprintf (stderr, "Data not available for bufr_file_out!\n");
        return 0;
    }

    /* element descriptor */

    if (des[ind]->id == ELDESC) {

        d = &(des[ind]->el->d);

        /* output descriptor if not inside a sequence */

        if (!in_seq && ind != ccitt_special && !_replicating 
            && ind != add_f_special)
            fprintf (fo_, "%2d %2d %3d ", d->f, d->x, d->y);

        /* descriptor without data (1.x.y, 2.x.y) or ascii) */

        if (ind == _desc_special) {

            unit = des[ind]->el->unit;

            /* special treatment for ASCII data */

            if (unit != NULL && strcmp (unit, "CCITT IA5") == 0) {
                fprintf (fo_, "       '");
                if (!bufr_parse_out (d, 0, 0, bufr_char_to_file, 0)) {
                    return 0;
                }
                fprintf (fo_, "'\n");
                nchars = des[ind]->el->dw / 8;                
            }

            /* only descriptor -> add newline */
            
            else if (!in_seq && !_replicating) {
                fprintf (fo_, "\n");
            }
        }

        /* "normal" data */

        else { 

            /* check for missing values and flag tables */

            if (val == MISSVAL) {
                strcpy (sval, "      missing");
            }
#if BUFR_OUT_BIN
            else if (desc_is_flagtable (ind)) {
                place_bin_values (val, ind, sval);
            }
#endif
            else {
                sprintf (sval, "%15.7f", val);
            }

            /* do we have a descriptor before the data element? */

            if (!in_seq && !_replicating && ind != add_f_special) {
                fprintf (fo_, "%s            %s\n", 
                         sval, des[ind]->el->elname);
            }
            else {
                if (!first_in_seq) 
                    fprintf (fo_, "          ");

                fprintf (fo_, "%s  %2d %2d %3d %s\n", 
                         sval, d->f, d->x, d->y, des[ind]->el->elname);
                first_in_seq = 0;
            }
        }
    } /* end if ("Element descriptor") */

    /* sequence descriptor */

    else if (des[ind]->id == SEQDESC) {

        d = &(des[ind]->seq->d);

        /* output descriptor if not inside another sequence descriptor */

        if (!in_seq && !_replicating)
            fprintf (fo_, "%2d %2d %3d ", d->f, d->x, d->y);

        /* check if bitmap or "normal" sequence descriptor */

        depth = check_bitmap_desc (d);

        /* seqdesc is a special opera bitmap */

        if (depth > 0) {

            strcpy (fname, imgfile_);

            /* Add the counter to the filename */
            
            if (count != 0) {
                sprintf (tmp, "%d", count);
                strcat (fname, tmp);
            }
            count ++;

            /* read bitmap and run length decode to file */

            vals = bufr_open_val_array ();
            if (vals == (bufrval_t*) NULL) return 0;

            _opera_mode = 1;
            if (!bufr_parse_out (des[ind]->seq->del, 0, des[ind]->seq->nel - 1,
                                 bufr_val_to_global, 0)) {
                _opera_mode = 0;
                bufr_close_val_array ();
                return 0;
            }
            _opera_mode = 0;
            nv = vals->nvals;

            if (depth == 8)
            {
                if (!z_decompress_to_file (fname, vals->vals, &nv)) 
                { 
                    bufr_close_val_array ();
                    fprintf (stderr, "Error during z-compression.\n");
                    return 0;
                }
            } else {

            /* Runlength decode */
            
                if (!rldec_to_file (fname, vals->vals, depth, &nv)) 
                { 
                    bufr_close_val_array ();
                    fprintf (stderr, "Error during runlength-compression.\n");
                    return 0;
                }
            }

            if (in_seq || _replicating) 
                fprintf (fo_, "        ");

            fprintf (fo_, "%s\n", fname);

            /* free array */

            bufr_close_val_array ();
            return 1;
        } 
        /* normal sequence descriptor - just call bufr_parse_out and
           remember that we are in a sequence */
        
        else {
            if (in_seq == 0)
                first_in_seq = 1;
            in_seq ++;
            ok = bufr_parse_out (des[ind]->seq->del, 0, 
                                 des[ind]->seq->nel - 1, bufr_file_out, 1);
            in_seq --;
            return ok;
        }
    } /* if ("seqdesc") */
    return 1;
}


/*===========================================================================*/
/** \ingroup cbout Outputs one character of an ASCII string to a file
    \brief Output one CCITT character to an ASCII file.

   This function outputs one CCITT (ASCII) character to a file which was
   opened by \ref bufr_open_output_file. 

   \param[in] val    Data-value to be output.
   \param[in] ind    Index to the global array \ref des[] holding the 
                     description of known data-descriptors.

   \return The function returns 1 on success, 0 on a fault.

   \see bufr_file_out, bufr_open_output_file, bufr_close_output_file
*/

static int bufr_char_to_file (varfl val, int ind)


{
    assert (ind == ccitt_special);

    if (fo_ == (FILE*) NULL) {
        fprintf (stderr, "Global file pointer not available!\n");
        return 0;
    }
    
    if (val == 0) val = 0x20;
    
    fprintf (fo_, "%c", (int) val);
    return 1;
}

/*===========================================================================*/
/** Reads next character from char_ and  stores position in cc_.

   \param[out] val The value of the character
   \param[in]  ind Index to the global array \ref des[] holding the 
               description of known data-descriptors.

   \return 1 on success, 0 on error.

   \see bufr_src_in


*/
static int bufr_input_char (varfl* val, int ind) {

    assert (ind == ccitt_special);

    if (char_ == NULL) {
        fprintf (stderr, "Global char pointer not available!\n");
        return 0;
    }
        
    /* check for correct string */
    
    if (*char_ != '\'') {
        fprintf (stderr, 
                 "Possible number of bits missmatch for ASCII data 1!\n");
        return 0;
    }

    /* check for correct number of characters */

    if (char_[cc_+1] == 0 || char_[cc_+1] == '\'') {
        fprintf (stderr, "Number of bits missmatch for ASCII data\n");
        return 0;
    }

    /* copy character to float */

    *val = (varfl) (unsigned char) char_[cc_+1];
    cc_++;


    return 1;

}

/*===========================================================================*/
/** Add one descriptor to array, allocate memory for array if necessary.
   Memory has to be freed by calling function!

   \param[in] d descriptor to be wrote
   \param[in,out] data The BUFR src structure containing the descriptor array

   \return 1 on success, 0 on error
 */

static int desc_to_array (dd* d, bufrsrc_t* data)

{
    int nd = data->ndesc;       /* number of data descriptors */
    dd* descs = data->descs;    /* array of data descriptors */

    if (nd > MAX_DESCS) {
        fprintf (stderr, "ERROR maximum number of descriptors exceeded!\n");
        return 0;
    }

    /* Allocate memory if not yet done */

    if (descs == (dd*) NULL) {
        descs = (dd *) malloc (MEMBLOCK * sizeof (dd));
        if (descs == (dd*) NULL) {
            fprintf (stderr, 
                     "Could not allocate memory for descriptor array!\n");
            return 0;
        }
		memset (descs, 0, MEMBLOCK * sizeof (dd));
        nd = 0;
    }

    /* Check if memory block is large enough to hold new data 
       and reallocate memory if not */

    if (nd != 0 && nd % MEMBLOCK == 0) {
        descs = (dd *) realloc (descs, (nd + MEMBLOCK) * sizeof (dd));
        if (descs == (dd*) NULL) {
            fprintf (stderr, 
                     "Could not reallocate memory for descriptor array!\n");
            return 0;
        }
		memset ((dd *) (descs + nd), 0, MEMBLOCK * sizeof (dd));
    }

    /* Add descriptor to array */

    memcpy ((dd*) (descs + nd), d, sizeof (dd));
    nd ++;
    data->ndesc = nd;
    data->descs = descs;
    return 1;
}

/*===========================================================================*/
/** Add one data string to array, allocate memory for array if necessary.
   Memory has to be freed by calling function!

   \param[in] s String to be wrote
   \param[in,out] bufr The BUFR src structure containing the data array

   \return 1 on success, 0 on error

 */

static int string_to_array (char* s, bufrsrc_t* bufr)

{
    int ns = bufr->ndat;        /* number of data elements */
    bd_t* data =  bufr->data;     /* array of data elements */
    

    if (ns > MAX_DATA) {
        fprintf (stderr, "ERROR maximum number of data elements exceeded!\n");
        return 0;
    }

    /* Allocate memory if not yet done */

    if (data == (bd_t*) NULL) {
        data = (bd_t*) malloc (MEMBLOCK * sizeof(bd_t));
        if (data == (bd_t*) NULL) {
            fprintf (stderr, 
                     "Could not allocate memory for data array!\n");
            return 0;
        }
		memset (data, 0, MEMBLOCK * sizeof(bd_t));
        ns = 0;
    }

    /* Check if memory block is large enough to hold new data 
       and reallocate memory if not */

    if (ns != 0 && ns % MEMBLOCK == 0) {
        data = (bd_t*) realloc (data, (ns + MEMBLOCK) * sizeof (bd_t));
        if (data == (bd_t*) NULL) {
            fprintf (stderr, 
                     "Could not reallocate memory for data array!\n");
            return 0;
        }
		memset ((bd_t*) (data + ns), 0, MEMBLOCK * sizeof (bd_t));
    }

    /* Add descriptor to array */

    data[ns] = str_save (s);
    ns ++;
    bufr->ndat = ns;
    bufr->data = data;
    return 1;
}
/*===========================================================================*/
/**
   Saves a string into a newly allocated memory area

   \param[in] str         pointer to the string.


   \return the pointer to the start of the string or NULL if a
           memory allocation error occured.
*/
static char *str_save(char *str)
{
    register char *p;
    int l;
    
    /* get the string size */
    
    l = strlen(str) + 1;

    /* allocate memory */
        
    p = malloc(l);
    if (p == NULL) {
        fprintf (stderr, 
                 "Could not allocate memory for string!\n");
        return NULL;
    }

    /* copy string into memory */
    
    memcpy(p, str, l);

    /* return pointer to string */
    
    return p;
}

/*===========================================================================*/
/** replaces binary values given as "b0101001101" by integers. 

    \param[in,out] buf String containing the value

*/

static void replace_bin_values (char *buf)



{
  char *p, *q, *r;
  int bin_val, v, i;
  
  p = buf;
  while ((p = strstr (p + 1, " b")) != NULL) {

    /* check if that is really a binary value and get the end the the beginning of that value */
  
    q = p + 2;
    bin_val = 1;
    while (*q != 0 && *q != ' ') {
      if (*q != '0' && *q != '1') bin_val = 0;
      q ++;
    }

    /* If it is a binary value convert it to an integer */

    if (bin_val) {
      r = q; r --;
      v = 0;
      for (i = 0; r >= p; i ++, r --) {
        if (*r == '1') v |= 1 << i;
      }

      /* Finally replace the binary data be the integer */

      sprintf (p + 1, "%d", v);
    }
  }
}
/*===========================================================================*/
/** Replaces an integer by its binary representation

    \param[in] val The value to be converted to binary format
    \param[in] ind Index to the global array \ref des[] holding the 
                     description of known data-descriptors.
    \param[in,out] buf  Pointer to the output string.

*/
#if BUFR_OUT_BIN
static void place_bin_values (varfl val, int ind, char* buf) {

    int dwi, i, bit;
    assert (buf != NULL);
    dwi = des[ind]->el->dw;

    strcpy (buf, "");
    for (i = 0; i < 13 - 1 - dwi; i ++) strcat (buf, " ");
    strcat (buf, "b");
    for (i = dwi - 1; i >= 0; i --) {
        if (1 << i & (long) val) {
            bit = 1;
        } else {
            bit = 0;
        }
        sprintf (buf + strlen (buf), "%d", bit);
    }
}
#endif

/*===========================================================================*/
/** Byte swap of 64bit values if host platform uses big endian

    \param[in,out] buf  buffer holding 64 bit values
    \param[in]     n    number of 64 bit values in buffer
*/
void byteswap64 (unsigned char *buf, int n)
{
    int i;
    unsigned char c;

    unsigned one = 1;
    unsigned char *test = (unsigned char *) &one;

    if (*test == 1)
        return;

    for (i = 0; i < n; i+= 8)
    {
        c = buf[0];
        buf[0] = buf[7];
        buf[7] = c;
        c = buf[1];
        buf[1] = buf[6];
        buf[6] = c;
        c = buf[2];
        buf[2] = buf[5];
        buf[5] = c;
        c = buf[3];
        buf[3] = buf[4];
        buf[4] = c;
        buf += 8;
    }
}

#include "zlib.h"
#define MAXBLOCK 65534

/*===========================================================================*/
/** z-decompression of array of bufr values with compressed bytes.
 *  Writes 64bit floats in platfrom native form to file 
 *  The float-bytes are swapped if the host representation is different
 *  from the IEEE byte order. 

    \param[in] outfile  Name of output file
    \param[in,out] vals    Array of compressed bytes stored as bufr values
    \param[in,out] nvals   Number of values in the array
    \return 1 for success, 0 on error
*/
int z_decompress_to_file (char* outfile, varfl* vals, int* nvals)
{
    FILE *fp;
    unsigned char *cbuf, *buf;
    int i, nv, ncols, nrows, sz;
    z_stream zs;
    
    memset (&zs, 0, sizeof(zs));
    fp = fopen (outfile, "wb");
    if (fp == NULL) {
        fprintf (stderr, "Could not open file %s!\n", outfile);
        return 0;
    }

    cbuf = malloc (MAXBLOCK);
    buf = malloc (MAXBLOCK);
    if (cbuf == NULL || buf == NULL)
    {
        fprintf (stderr, "malloc error\n");
        if (cbuf != NULL) free (cbuf);
        if (buf != NULL) free (buf);
        return 0;
    }
    
    inflateInit (&zs);
    
    sz = 0;
    nv = 0;
    nv++;
    nrows = vals[nv++];
    while (nrows-- > 0)
    {
        ncols = vals[nv++];
        for (i = 0; i < ncols; i++)
        {
            cbuf[i] = (unsigned char) vals[nv++];
        }
        zs.next_in = cbuf;
        zs.avail_in = ncols;
        while (zs.avail_in > 0)
        {
            int err;
            zs.next_out = buf + sz;
            zs.avail_out = MAXBLOCK - sz;
            err = inflate (&zs, Z_SYNC_FLUSH);
            if (err != Z_OK && err != Z_STREAM_END)
                break;

            sz = (MAXBLOCK - zs.avail_out) / 8 * 8;
            byteswap64 (buf, sz); 
            fwrite (buf, 1, sz, fp);
            sz = MAXBLOCK - zs.avail_out - sz;
            if (sz > 0)
                memmove (buf, buf + MAXBLOCK - zs.avail_out - sz, sz);
        }
    }

    inflateEnd (&zs);
    free (buf);
    free (cbuf);
    fclose(fp);
    *nvals = nv;
    return 1;
}


/*===========================================================================*/
/** Reads 64bit floats in platfrom native form from file, apllies 
 *  z-compression and puts the compressed bytes as bufr values in the array.
 *  The float-bytes are swapped if the host representation is different
 *  from the IEEE byte order. 

    \param[in] infile  Name of input file
    \param[in,out] vals    Array of compressed bytes stored as bufr values
    \param[in,out] nvals   Number of values in the array
    \return 1 for success, 0 on error
*/
int z_compress_from_file (char* infile, varfl* *vals, int* nvals)
{
    FILE *fp;
    int nv, sz, n;
    unsigned char *buf, *cbuf;
    unsigned long sz1;
    
    fp = fopen (infile, "rb");
    if (fp == NULL) {
        fprintf (stderr, "error opening '%s'\n", infile);
        return 0;
    }

    fseek (fp, 0, SEEK_END);
    sz = ftell (fp);
    
    if ((buf = malloc (sz)) == NULL)
    {
        fclose (fp);
        fprintf (stderr, "malloc error\n");
        return 0;
    }
    
    fseek (fp, 0, SEEK_SET);
    if (fread (buf, 1, sz, fp) != sz)
    {
        fclose (fp);
        free (buf);
        fprintf (stderr, "read error\n");
        return 0;
    }
    fclose (fp);

    byteswap64 (buf, sz);
    
    sz1 = sz + sz / 1000 + 100 + 12;
    if ((cbuf = malloc (sz1)) == NULL)
    {
        free (buf);
        fprintf (stderr, "malloc error\n");
        return 0;
    }
    if (compress (cbuf, &sz1, buf, sz) != Z_OK)
    {
        free (buf);
        free (cbuf);
        fprintf (stderr, "compress error\n");
        return 0;
    }
    
    free (buf);
    buf = cbuf;
    sz = sz1;
    nv = (sz + MAXBLOCK - 1) / (MAXBLOCK);
    
    bufr_val_to_array (vals, 0, nvals);
    bufr_val_to_array (vals, nv, nvals);

    while (nv-- > 0)
    {
        n = sz < MAXBLOCK ? sz : MAXBLOCK;
        bufr_val_to_array (vals, n, nvals);
        while (n-- > 0)
        {
            if (bufr_val_to_array (vals, *buf++, nvals) == 0)
                break;
            sz--;
        }
        if (n > 0)
            break;
    }
    free (cbuf);
    return sz == 0;
}


/*===========================================================================*/
/** z-decompression of array of bufr values with compressed bytes.
 *  Writes 64bit floats in platfrom native to a memory area. 
 *  The float-bytes are swapped if the host representation is different
 *  from the IEEE byte order. 

    \param[in,out] data  Pointer to receive data array
    \param[in,out] vals    Array of compressed bytes stored as bufr values
    \param[in,out] nvals   Number of values in the array
    \return number of data values or 0 on error
*/
int bufr_z_decompress_to_mem (varfl **data, varfl* vals, int* nvals)
{
    unsigned char *cbuf, *buf, *outbuf;
    int i, nv, ncols, nrows, sz, out_size, out_used;
    z_stream zs;
    
    memset (&zs, 0, sizeof(zs));

    cbuf = malloc (MAXBLOCK);
    buf = malloc (MAXBLOCK);
    outbuf = malloc (MAXBLOCK);
    out_size = MAXBLOCK;
    out_used = 0;
    
    if (cbuf == NULL || buf == NULL || outbuf == NULL)
    {
        fprintf (stderr, "malloc error\n");
        if (cbuf != NULL) free (cbuf);
        if (buf != NULL) free (buf);
        if (outbuf != NULL) free (outbuf);
        return 0;
    }
    
    inflateInit (&zs);
    
    sz = 0;
    nv = 0;
    nv++;
    nrows = vals[nv++];
    while (nrows-- > 0)
    {
        ncols = vals[nv++];
        for (i = 0; i < ncols; i++)
        {
            cbuf[i] = (unsigned char) vals[nv++];
        }
        zs.next_in = cbuf;
        zs.avail_in = ncols;
        while (zs.avail_in > 0)
        {
            int err;
            zs.next_out = buf + sz;
            zs.avail_out = MAXBLOCK - sz;
            err = inflate (&zs, Z_SYNC_FLUSH);
            if (err != Z_OK && err != Z_STREAM_END)
                break;

            sz = (MAXBLOCK - zs.avail_out) / 8 * 8;
            byteswap64 (buf, sz);
            if (out_size - out_used < sz)
            {
                out_size = 2 * (out_size + MAXBLOCK);
                outbuf = realloc (outbuf, out_size);
                if (outbuf == NULL)
                {
                    inflateEnd (&zs);
                    free (cbuf);
                    free (buf);
                    return 0;
                }
            }
            
            memcpy (outbuf + out_used, buf, sz);
            out_used += sz;
            sz = MAXBLOCK - zs.avail_out - sz;
            if (sz > 0)
                memmove (buf, buf + MAXBLOCK - zs.avail_out - sz, sz);
        }
    }

    inflateEnd (&zs);
    free (buf);
    free (cbuf);
    *nvals = nv;
    *data = (varfl *) outbuf;
    return out_used / 8;
}

/*===========================================================================*/
/** Reads 64bit floats in platfrom native form from file, apllies 
 *  z-compression and puts the compressed bytes as bufr values in the array.
 *  The float-bytes are swapped if the host representation is different
 *  from the IEEE byte order. 

    \param[in] data  Array of data elements
    \param[in] ndata  Number of data elements
    \param[in,out] vals    Array of compressed bytes stored as bufr values
    \param[in,out] nvals   Number of values in the array
    \return 1 for success, 0 on error
*/
int bufr_z_compress_from_mem (varfl *data, int ndata, varfl* *vals, int* nvals)
{
    int nv, sz, n;
    unsigned char *buf, *cbuf;
    unsigned long sz1;
    
    sz = 8 * ndata;
    buf = (unsigned char *) data;
    byteswap64 (buf, sz);
    
    sz1 = sz + sz / 1000 + 100 + 12;
    if ((cbuf = malloc (sz1)) == NULL)
    {
        byteswap64 (buf, sz);
        fprintf (stderr, "malloc error\n");
        return 0;
    }
    if (compress (cbuf, &sz1, buf, sz) != Z_OK)
    {
        byteswap64 (buf, sz);
        free (cbuf);
        fprintf (stderr, "compress error\n");
        return 0;
    }
    
    byteswap64 (buf, sz);
    buf = cbuf;
    sz = sz1;
    nv = (sz + MAXBLOCK - 1) / (MAXBLOCK);
    
    bufr_val_to_array (vals, 0, nvals);
    bufr_val_to_array (vals, nv, nvals);

    while (nv-- > 0)
    {
        n = sz < MAXBLOCK ? sz : MAXBLOCK;
        bufr_val_to_array (vals, n, nvals);
        while (n-- > 0)
        {
            if (bufr_val_to_array (vals, *buf++, nvals) == 0)
                break;
            sz--;
        }
        if (n > 0)
            break;
    }
    free (cbuf);
    return sz == 0;
}


#define VV(i) ((i - 50000)/100.0)
void z_test()
{
    bufrval_t* vals;
    varfl v, *data;
    int i, n, nvals;
    FILE *f;
    
    f = fopen ("test.1", "w");
    for (i = 0; i < 100000; i++)
    {
        v = VV(i);
        fwrite (&v, sizeof(v), 1, f);
    }
    fclose(f);

    vals = bufr_open_val_array();
    z_compress_from_file ("test.1", &vals->vals, &vals->nvals);
    n = bufr_z_decompress_to_mem (&data, vals->vals, &nvals);
    bufr_close_val_array();

    vals = bufr_open_val_array();
    bufr_z_compress_from_mem (data, n, &vals->vals, &nvals);
    z_decompress_to_file ("test.2", vals->vals, &nvals);
    bufr_close_val_array();

    f = fopen ("test.2", "w");
    fwrite (data, 8, n, f);
    fclose (f);
    free(data);
  
    f = fopen ("test.2", "r");
    for (i = 0; i < 100000; i++)
    {
        fread (&v, sizeof(v), 1, f);
        if (v != VV(i))
            printf ("%6d: %12.6f %12.6f\n", i, v, VV(i));
    }
    fclose(f);
}

/* end of file */



/* APISAMPLE.c */


/*===========================================================================*/
/* internal function definitons                                              */
/*===========================================================================*/

static void create_source_msg (dd* descs, int* nd, varfl** vals, 
                               radar_data_t* d);
static int our_callback (varfl val, int ind);
static void create_sample_data (radar_data_t* d);

/*===========================================================================*/
/* internal data                                                             */
/*===========================================================================*/

radar_data_t our_data; /* sturcture holding our decoded data */
char *version = "apisample V3.0, 5-Dec-2007\n";

/*===========================================================================*/



/** \ingroup samples
    \brief Sample for encoding a BUFR message.

    This function encodes sample data to a BUFR message and saves the
    results to a file apisample.bfr, also returns the encoded message.

    \param[in]  src_data Our source data.
    \param[out] bufr_msg Our encoded BUFR message.

    \see bufr_decoding_sample

*/

void bufr_encoding_sample (radar_data_t* src_data, bufr_t* bufr_msg) {

    sect_1_t s1;          /* structure holding information from section 1 */
    dd descs[MAX_DESCS];  /* array of data descriptors, must be large enough
                             to hold all required descriptors */
    int nd = 0;           /* current number of descriptors in descs */
    varfl* vals = NULL;   /* array of data values */
    int ok;

    long year, mon, day, hour, min;

    memset (&s1, 0, sizeof (sect_1_t));
    
    /* first let's create our source message */

    create_source_msg (descs, &nd, &vals, src_data);

    /* Prepare data for section 1 */

    s1.year = 999;
    s1.mon  = 999;
    s1.day = 999;
    s1.hour = 999;
    s1.min  = 999;
    s1.mtab = 0;                      /* master table used */
    s1.subcent = 255;                 /* originating subcenter */
    s1.gencent = 255;                 /* originating center */
    s1.updsequ = 0;                   /* original BUFR message */
    s1.opsec = 0;                     /* no optional section */
    s1.dcat = 6;                      /* message type */
    s1.dcatst = 0;                    /* message subtype */
    s1.vmtab = 11;                    /* version number of master table used */
    s1.vltab = 4;                     /* version number of local table used */

    /* read supported data descriptors from tables */

    ok = (read_tables (NULL, s1.vmtab, s1.vltab, s1.subcent, s1.gencent) >= 0);

    /* encode our data to a data-descriptor- and data-section */

    if (ok) ok = bufr_encode_sections34 (descs, nd, vals, bufr_msg);

    /* setup date and time if necessary */

    if (ok && s1.year == 999) {
        bufr_get_date_time (&year, &mon, &day, &hour, &min);
        s1.year = (int) year;
        s1.mon = (int) mon;
        s1.day = (int) day;
        s1.hour = (int) hour;
        s1.min = (int) min;
        s1.sec = 0;
    }

    /* encode section 0, 1, 2, 5 */

    if (ok) ok = bufr_encode_sections0125 (&s1, bufr_msg);

    /* Save coded data */

    if (ok) ok = bufr_write_file (bufr_msg, "apisample.bfr");

    if (vals != NULL)
        free (vals);
    free_descs ();

    if (!ok) exit (EXIT_FAILURE);
}

/*===========================================================================*/
/** \ingroup samples
    \brief Sample for decoding a BUFR message.

    This function decodes a BUFR message and stores the values in
    our sample radar data structure. Also saves the result to a file.

    \param[in] msg Our encoded BUFR message.
    \param[out] data Our source data.

    \see bufr_encoding_sample
*/

void bufr_decoding_sample (bufr_t* msg, radar_data_t* data) {

    sect_1_t s1;
    int ok, desch, ndescs, subsets;
    dd* dds = NULL;

    /* initialize variables */

    memset (&s1, 0, sizeof (sect_1_t));

    /* Here we could also read our BUFR message from a file */
    /* bufr_read_file (msg, buffile); */

    /* decode section 1 */

    ok = bufr_decode_sections01 (&s1, msg);

    /* Write section 1 to ASCII file */

    bufr_sect_1_to_file (&s1, "section.1.out");

    /* read descriptor tables */

    if (ok) ok = (read_tables (NULL, s1.vmtab, s1.vltab, s1.subcent, 
                               s1.gencent) >= 0);

    /* decode data descriptor and data-section now */

    /* open bitstreams for section 3 and 4 */

    desch = bufr_open_descsec_r(msg, &subsets);
    ok = (desch >= 0);
    if (ok) ok = (bufr_open_datasect_r(msg) >= 0);

    /* calculate number of data descriptors  */
    
    ndescs = bufr_get_ndescs (msg);

    /* allocate memory and read data descriptors from bitstream */

    if (ok) ok = bufr_in_descsec (&dds, ndescs, desch);

    /* output data to our global data structure */

    while (ok && subsets--) 
        ok = bufr_parse_out (dds, 0, ndescs - 1, our_callback, 1);

    /* get data from global */

    data = &our_data;

    /* close bitstreams and free descriptor array */

    if (dds != (dd*) NULL)
        free (dds);
    bufr_close_descsec_r (desch);
    bufr_close_datasect_r ();

    /* decode data to file also */

    if (ok) ok = bufr_data_to_file ("apisample.src", "apisample.img", msg);

    bufr_free_data (msg);
    free_descs();
    exit (EXIT_SUCCESS);


}

/*===========================================================================*/



/*===========================================================================*/
#define fill_desc(ff,xx,yy) {\
        dd.f=ff; dd.x=xx; dd.y=yy; \
        bufr_desc_to_array (descs, dd, nd);}
#define fill_v(val) bufr_val_to_array (vals, val, &nv);

/**
   create our source BUFR message according to the OPERA BUFR guidelines 
*/
static void create_source_msg (dd* descs, int* nd, varfl** vals, 
                               radar_data_t* d) {

    dd dd;
    int nv = 0, i;

    fill_desc(3,1,1);           /* WMO block and station number */
    fill_v(d->wmoblock);
    fill_v(d->wmostat);

    fill_desc(3,1,192);         /* Meta information about the product */
    fill_v(d->meta.year);       /* Date */
    fill_v(d->meta.month);
    fill_v(d->meta.day);
    fill_v(d->meta.hour);       /* Time */
    fill_v(d->meta.min);
    fill_v(d->img.nw.lat);      /* Lat. / lon. of NW corner */
    fill_v(d->img.nw.lon);
    fill_v(d->img.ne.lat);      /* Lat. / lon. of NE corner */
    fill_v(d->img.ne.lon);
    fill_v(d->img.se.lat);      /* Lat. / lon. of SE corner */
    fill_v(d->img.se.lon);
    fill_v(d->img.sw.lat);      /* Lat. / lon. of SW corner */
    fill_v(d->img.sw.lon);
    fill_v(d->proj.type);             /* Projection type */
    fill_v(d->meta.radar.lat);        /* Latitude of radar */
    fill_v(d->meta.radar.lon);        /* Longitude of radar */
    fill_v(d->img.psizex);            /* Pixel size along x coordinate */
    fill_v(d->img.psizey);            /* Pixel size along y coordinate */
    fill_v(d->img.nrows);             /* Number of pixels per row */
    fill_v(d->img.ncols);             /* Number of pixels per column */

    fill_desc(3,1,22);          /* Latitude, longitude and height of station */
    fill_v(d->meta.radar.lat);
    fill_v(d->meta.radar.lon);
    fill_v(d->meta.radar_height);

                                /* Projection information (this will be 
                                   a sequence descriptor when using tables 6 */
    fill_desc(0,29,199);        /* Semi-major axis or rotation ellipsoid */
    fill_v(d->proj.majax);
    fill_desc(0,29,200);        /* Semi-minor axis or rotation ellipsoid */
    fill_v(d->proj.minax);
    fill_desc(0,29,193);        /* Longitude Origin */
    fill_v(d->proj.orig.lon);
    fill_desc(0,29,194);        /* Latitude Origin */
    fill_v(d->proj.orig.lat);
    fill_desc(0,29,195);        /* False Easting */
    fill_v(d->proj.xoff);
    fill_desc(0,29,196);        /* False Northing */
    fill_v(d->proj.yoff);
    fill_desc(0,29,197);        /* 1st Standard Parallel */
    fill_v(d->proj.stdpar1);
    fill_desc(0,29,198);        /* 2nd Standard Parallel */
    fill_v(d->proj.stdpar2);

    fill_desc(0,30,31);         /* Image type */
    fill_v(d->img.type);

    fill_desc(0,29,2);          /* Co-ordinate grid */
    fill_v(d->img.grid);

    fill_desc(0,33,3);          /* Quality information */
    fill_v(d->img.qual);


    /* level slicing table note the use of change of datawith in order to 
       encode our values, also values are converted to integer, loosing
       precision
    */

    fill_desc(2,1,129);          /* change of datawidth because 0 21 1 
                                    only codes to 7 bit */
    fill_desc(3,13,9);          /* Reflectivity scale */
    fill_v(d->img.scale.vals[0]);   /* scale[0] */
    fill_v(d->img.scale.nvals -1);     /* number of scale values - 1 */ 
    for (i = 1; i < d->img.scale.nvals; i++) {
        fill_v(d->img.scale.vals[i]);
    }
    fill_desc(2,1,0);          /* cancel change of datawidth */

    /* another possibility for the level slicing table withour using
       datawidth and scale change and without loosing precision */

    fill_desc(0,21,198);        /* dBZ Value offset */
    fill_v(d->img.scale.offset);
    fill_desc(0,21,199);        /* dBZ Value increment */
    fill_v(d->img.scale.increment);


    fill_desc(3,21,193);        /* 8 bit per pixel pixmap */

    /* run length encode our bitmap */
    rlenc_from_mem (d->img.data, d->img.nrows, d->img.ncols, vals, &nv);
    
    free(d->img.data);
}

/*===========================================================================*/

/** Our callback for storing the values in our radar_data_t structure 
    and for run-length decoding the radar image 
*/

static int our_callback (varfl val, int ind) {

    radar_data_t* b = &our_data;   /* our global data structure */
    bufrval_t* v;                  /* array of data values */
    varfl* vv;
    int i = 0, nv, nr, nc;
    dd* d;

    /* do nothing if data modifictaon descriptor or replication descriptor */
    if (ind == _desc_special) return 1;

    /* sequence descriptor */
    if (des[ind]->id == SEQDESC) {

        /* get descriptor */

        d = &(des[ind]->seq->d);

        /* open array for values */

        v = bufr_open_val_array ();
        if (v == (bufrval_t*) NULL) return 0;

        /* WMO block and station number */

        if  (bufr_check_fxy (d, 3,1,1)) {   

            /* decode sequence to global array */

            bufr_parse_out (des[ind]->seq->del, 0, des[ind]->seq->nel - 1,
                            bufr_val_to_global, 0);

            /* get our data from the array */

            b->wmoblock = (int) v->vals[i++];
            b->wmostat = (int) v->vals[i];

        }
        /* Meta information */

        else if (bufr_check_fxy (d, 3,1,192)) { 

            bufr_parse_out (des[ind]->seq->del, 0, des[ind]->seq->nel - 1,
                            bufr_val_to_global, 0);
            vv = v->vals;
            i = 0;
            b->meta.year = (int) vv[i++];       /* Date */
            b->meta.month = (int) vv[i++];
            b->meta.day = (int) vv[i++];
            b->meta.hour = (int) vv[i++];       /* Time */
            b->meta.min = (int) vv[i++];
            b->img.nw.lat = vv[i++];      /* Lat. / lon. of NW corner */
            b->img.nw.lon = vv[i++];
            b->img.ne.lat = vv[i++];      /* Lat. / lon. of NE corner */
            b->img.ne.lon = vv[i++];
            b->img.se.lat = vv[i++];      /* Lat. / lon. of SE corner */
            b->img.se.lon = vv[i++];
            b->img.sw.lat = vv[i++];      /* Lat. / lon. of SW corner */
            b->img.sw.lon = vv[i++];
            b->proj.type = (int) vv[i++];       /* Projection type */
            b->meta.radar.lat = vv[i++];        /* Latitude of radar */
            b->meta.radar.lon = vv[i++];        /* Longitude of radar */
            b->img.psizex = vv[i++];      /* Pixel size along x coordinate */
            b->img.psizey = vv[i++];      /* Pixel size along y coordinate */
            b->img.nrows = (int) vv[i++];     /* Number of pixels per row */
            b->img.ncols = (int) vv[i++];     /* Number of pixels per column */

        }
        /* Latitude, longitude and height of station */

        else if (bufr_check_fxy (d, 3,1,22)) { 

            bufr_parse_out (des[ind]->seq->del, 0, des[ind]->seq->nel - 1,
                            bufr_val_to_global, 0);
            vv = v->vals;
            i = 0;
            b->meta.radar.lat = vv[i++];
            b->meta.radar.lon = vv[i++];
            b->meta.radar_height = vv[i];
        }
        /* Reflectivity scale */

        else if (bufr_check_fxy (d, 3,13,9)) { 
            int j;

            bufr_parse_out (des[ind]->seq->del, 0, des[ind]->seq->nel - 1,
                            bufr_val_to_global, 0);
            vv = v->vals;
            i = 0;
            
            b->img.scale.vals[0] = vv[i++];
            b->img.scale.nvals = (int) vv[i++] + 1;  /* number of scale values */ 
            assert(b->img.scale.nvals < 256);
            for (j = 1; j < b->img.scale.nvals; j++) {
                b->img.scale.vals[j] = vv[i++];
            }
        }

        /* our bitmap */

        else if (bufr_check_fxy (d, 3,21,193)) {

            /* read bitmap and run length decode */

            if (!bufr_parse_out (des[ind]->seq->del, 0, des[ind]->seq->nel - 1,
                                 bufr_val_to_global, 0)) {
                bufr_close_val_array ();
                return 0;
            }

            if (!rldec_to_mem (v->vals, &(b->img.data), &nv, &nr, &nc)) { 
                bufr_close_val_array ();
                fprintf (stderr, "Error during runlength-compression.\n");
                return 0;
            }
        }
     
        else {
            fprintf (stderr,
                     "Unknown sequence descriptor %d %d %d", d->f, d->x, d->y);
        }
        /* close the global value array */

        bufr_close_val_array ();

    }

    /* element descriptor */

    else if (des[ind]->id == ELDESC) {

        d = &(des[ind]->el->d);

        if (bufr_check_fxy (d, 0,29,199))
            /* Semi-major axis or rotation ellipsoid */
            b->proj.majax = val;
        else if (bufr_check_fxy (d, 0,29,200))
            /* Semi-minor axis or rotation ellipsoid */
            b->proj.minax = val;
        else if (bufr_check_fxy (d, 0,29,193))
            /* Longitude Origin */
            b->proj.orig.lon = val;
        else if (bufr_check_fxy (d, 0,29,194))
            /* Latitude Origin */
            b->proj.orig.lat = val;
        else if (bufr_check_fxy (d, 0,29,195))
            /* False Easting */
            b->proj.xoff = (int) val;
        else if (bufr_check_fxy (d, 0,29,196))
            /* False Northing */
            b->proj.yoff = (int) val;
        else if (bufr_check_fxy (d, 0,29,197))
            /* 1st Standard Parallel */
            b->proj.stdpar1 = val;
        else if (bufr_check_fxy (d, 0,29,198))
            /* 2nd Standard Parallel */
            b->proj.stdpar2 = val;
        else if (bufr_check_fxy (d, 0,30,31))
            /* Image type */
            b->img.type = (int) val;
        else if (bufr_check_fxy (d, 0,29,2))
            /* Co-ordinate grid */
            b->img.grid = (int) val;
        else if (bufr_check_fxy (d, 0,33,3))
            /* Quality information */
            b->img.qual = val;
        else if (bufr_check_fxy (d, 0,21,198))
            /* dBZ Value offset */
            b->img.scale.offset = val;
        else if (bufr_check_fxy (d, 0,21,199))
            /* dBZ Value increment */
            b->img.scale.increment = val;
        else {
            fprintf (stderr,
                     "Unknown element descriptor %d %d %d", d->f, d->x, d->y);
            return 0;
        }
    }
    return 1;
}

/*===========================================================================*/
#define NROWS 200   /* Number of rows for our sample radar image */
#define NCOLS 200   /* Number of columns for our sample radar image */

static void create_sample_data (radar_data_t* d) {

    int i;

    /* create a sample radar image */
    
    d->img.data = (unsigned short*) calloc (NROWS * NCOLS, 
                                            sizeof (unsigned short));

    if (d->img.data == NULL) {
        fprintf (stderr, "Could not allocate memory for sample image!\n");
        exit (EXIT_FAILURE);
    }

    /* fill image with random data (assuming 8 bit image depth -> max
       value = 254; 255 is missing value) */

#ifdef VERBOSE
    fprintf (stderr, "RAND_MAX = %d\n", RAND_MAX);
#endif

    for (i = 0; i < NROWS * NCOLS; i++) {
        d->img.data[i] = (unsigned short) ((float) rand() / RAND_MAX * 254);
#ifdef VERBOSE
        fprintf (stderr, "Value: %d\n", d->img.data[i]);
#endif
    }
    
    /* create our source data */

    d->wmoblock = 11;
    d->wmostat  = 164;

    d->meta.year = 2007;
    d->meta.month = 12;
    d->meta.day = 5;
    d->meta.hour = 12;
    d->meta.min = 5;
    d->meta.radar.lat = 47.06022;
    d->meta.radar.lon = 15.45772;
    d->meta.radar_height = 355;

    d->img.nw.lat = 50.4371;
    d->img.nw.lon = 8.1938;
    d->img.ne.lat = 50.3750;
    d->img.ne.lon = 19.7773;
    d->img.se.lat = 44.5910;
    d->img.se.lon = 19.1030;
    d->img.sw.lat = 44.6466;
    d->img.sw.lon = 8.7324;
    d->img.psizex = 1000;
    d->img.psizey = 1000;
    d->img.nrows = NROWS;
    d->img.ncols = NCOLS;
    d->img.type = 2;
    d->img.grid = 0;
    d->img.qual = MISSVAL;

    /* create level slicing table */

    d->img.scale.nvals = 255;

    for (i = 0; i < 255; i++) {
        d->img.scale.vals[i] = i * 0.5 - 31.0;
    }
    d->img.scale.offset = -31;
    d->img.scale.increment = 0.5;

    d->proj.type = 2;
    d->proj.majax = 6378137;
    d->proj.minax = 6356752;
    d->proj.orig.lon = 13.333333;
    d->proj.orig.lat = 47.0;
    d->proj.xoff = 458745;
    d->proj.yoff = 364548;
    d->proj.stdpar1 = 46.0;
    d->proj.stdpar2 = 49.0;
}

/** \example apisample.c

    This is an example for encoding and decoding a BUFR massage.\n
*/

/* end of file */





int main (int argc, char* argv[]) {

    bufr_t bufr_msg ;   /* structure holding encoded bufr message */

    /* initialize variables */
    memset (&bufr_msg, 0, sizeof (bufr_t));
    memset (&our_data, 0, sizeof (radar_data_t));

	if (argc < 1)
	return 0;

    int i;
    int strsize = 0;
    for (i=1; i<argc; i++) {
        strsize += strlen(argv[i]);
        if (argc > i+1)
            strsize++;
    }

    /*printf("strsize: %d\n", strsize);*/

    char *cmdstring;
    cmdstring = malloc(strsize);
    cmdstring[0] = '\0';

    for (i=1; i<argc; i++) {
        strcat(cmdstring, argv[i]);
        if (argc > i+1)
            strcat(cmdstring, " ");
    }

    printf("cmdstring: %s\n", cmdstring);
	
	
	

	/* Leemos Archivo BUFR */
	/*bufr_read_file (&bufr_msg, &buffile);*/
	bufr_read_file (&bufr_msg, cmdstring);

    /* sample for decoding from BUFR */
    bufr_decoding_sample (&bufr_msg, &our_data);
    bufr_free_data (&bufr_msg);

    /*free (our_data.img.data);*/

    exit (EXIT_SUCCESS);
}



