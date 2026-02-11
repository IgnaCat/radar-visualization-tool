/*-------------------------------------------------------------------------

    BUFR encoding and decoding software and library
    Copyright (c) 2007,  Institute of Broadband Communication, TU-Graz
    on behalf of EUMETNET OPERA, http://www.knmi.nl/opera

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; version 2.1 
    of the License.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA 

----------------------------------------------------------------------------

FILE:          DECBUFR.C
IDENT:         $Id: decbufr.c,v 1.18 2012/11/28 18:41:48 helmutp Exp $

AUTHORS:       Juergen Fuchsberger, Helmut Paulitsch, Konrad Koeck
               Institute of Communication and Wave Propagation, 
               Technical University Graz, Austria

VERSION NUMBER:3.2

DATE CREATED:  18-DEC-2001

STATUS:        DEVELOPMENT FINISHED
--------------------------------------------------------------------------- */

/** \file decbufr.c
    \brief Reads a BUFR-file, decodes it and stores decoded data in a
    text-file.

    This function reads a BUFR-file, decodes it and stores decoded data in a
    text-file. Decoded bitmaps are stored in a seperate file.

*/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "bufrlib.h"
#include "bufr_io.h"
#include "apisample.h"

static int our_callback (varfl val, int ind);

radar_data_t our_data; /* sturcture holding our decoded data */

int flag_data = 0;
int flag_size = 0;
int sweep_data[15000000];
int size_data;
double* vv;

/*===========================================================================*/
/* internal functions                                                        */
/*===========================================================================*/

#define RIOUTFILE  "img.dec"    /* Name of file for uncompressed radar image */

char *usage = 
"Usage: decbufr [-v] [-d tabdir] [-s1 sect1] input_file output_file [image_file]\n"
"       decbufr -show [-m mtab] [-l ltab] [-o ocent] [-s subcent] [f x y]\n";
char *version = "decbufr V3.2, 28-November-2012\n";

/*===========================================================================*/


int main (int argc, char** argv)

{
    char destfile[200], buffile[200];
    char *table_dir = NULL;
    char *sect1_file = "section.1.out";
    char imgfile[200];  /* filename of uncompressed image */
    sect_1_t s1;
    bufr_t bufr_msg;    /* structure holding encoded bufr message */

    /* initialize variables */

    memset (&bufr_msg, 0, sizeof (bufr_t));
    memset (&s1, 0, sizeof (sect_1_t));

    /* check command line parameter */

    while (argc > 1 && *argv[1] == '-')
    {
        if (*(argv[1] + 1) == 'v')
            fprintf (stderr, "%s", version);
        else if (*(argv[1] + 1) == 'd')
        {
            if (argc < 2)
            {
                fprintf (stderr, "Missing parameter for -d\n\n%s", usage);
                exit (EXIT_FAILURE);
            }
            table_dir = argv[2];
            argc--;
            argv++;
        }
        else if (strcmp (argv[1], "-s1") == 0)
        {
            sect1_file = argv[2];
            argc--; 
            argv++;
        }
        else if (strcmp (argv[1], "-show") == 0)
        {
            show_desc_args (argc - 1, argv + 1);
            exit (0);
        }
        else
        {
            fprintf (stderr, "Invalid parameter %s\n\n%s", argv[1], usage);
            exit (EXIT_FAILURE);
        }
        argc--;
        argv++;
    }

    /* Get input- and output-filenames from the command-line */

    if (argc < 3)
    {
        fprintf (stderr, "%s", usage);
        exit (EXIT_FAILURE);
    }
    strcpy (buffile, argv[1]);
    strcpy (destfile, argv[2]);

    if (argc > 3) 
        strcpy (imgfile, argv[3]);
    else 
        strcpy (imgfile, RIOUTFILE);

    /* read source-file. Therefore allocate memory to hold the complete
       BUFR-message */

    if (!bufr_read_file (&bufr_msg, buffile)) {
        bufr_free_data (&bufr_msg);
        exit (EXIT_FAILURE);
    }

    /* decode section 1 */

    if (!bufr_decode_sections01 (&s1, &bufr_msg)) {
        bufr_free_data (&bufr_msg);
        exit (EXIT_FAILURE);
    }

    /* Write section 1 to ASCII file */

    if (!bufr_sect_1_to_file (&s1, sect1_file)) {
        bufr_free_data (&bufr_msg);
        exit (EXIT_FAILURE);
    }

    /* read descriptor tables */

    if (read_tables (table_dir, s1.vmtab, s1.vltab, s1.subcent, 
                     s1.gencent) < 0) {
        bufr_free_data (&bufr_msg);
        free_descs();
        exit (EXIT_FAILURE);
    }

 
    /* decode data descriptor- and data-section now */
    if (!bufr_data_to_file (destfile, imgfile, &bufr_msg)) {
        fprintf (stderr, "unable to decode BUFR-message !\n");
        bufr_free_data (&bufr_msg);
        free_descs();
        exit (EXIT_FAILURE);
    }
    
    

#ifdef VERBOSE
    {
        int i;
        for (i = 0; i < 6; i++) {
            fprintf (stderr, "section %d length = %d\n", i, bufr_msg.secl[i]);
        }
    }
#endif

    bufr_free_data (&bufr_msg);
    free_descs();
    exit (8);
}






meta_t* get_meta_data (char* buffile, char* table_dir)
{
    /*char *table_dir = NULL;*/
    char *sect1_file = "section.9.out";
    char imgfile[200];  /* filename of uncompressed image */
    sect_1_t s1;
    bufr_t bufr_msg;    /* structure holding encoded bufr message */
	int ok, desch, ndescs, subsets;
    dd* dds = NULL;


    /* initialize variables */
    memset (&bufr_msg, 0, sizeof (bufr_t));
    memset (&s1, 0, sizeof (sect_1_t));
    memset (&our_data, 0, sizeof (radar_data_t));


    /* read source-file. Therefore allocate memory to hold the complete
       BUFR-message */
    if (!bufr_read_file (&bufr_msg, buffile)) {
        bufr_free_data (&bufr_msg);
        exit (EXIT_FAILURE);    }

    /* decode section 1 */
    if (!bufr_decode_sections01 (&s1, &bufr_msg)) {
        bufr_free_data (&bufr_msg);
        exit (EXIT_FAILURE);    }

    /* Write section 1 to ASCII file 
    if (!bufr_sect_1_to_file (&s1, sect1_file)) {
        bufr_free_data (&bufr_msg);
        exit (EXIT_FAILURE);    }*/

    /* read descriptor tables */
    if (read_tables (table_dir, s1.vmtab, s1.vltab, s1.subcent, 
                     s1.gencent) < 0) {
        bufr_free_data (&bufr_msg);
        free_descs();
        exit (EXIT_FAILURE);    }


    /* decode data descriptor and data-section now */
    /* open bitstreams for section 3 and 4 */
    desch = bufr_open_descsec_r(&bufr_msg, &subsets);
    ok = (desch >= 0);       
    if (ok) ok = (bufr_open_datasect_r(&bufr_msg) >= 0);
	
    /* calculate number of data descriptors  */  
    ndescs = bufr_get_ndescs (&bufr_msg);
    
    /* allocate memory and read data descriptors from bitstream */
    if (ok) ok = bufr_in_descsec (&dds, ndescs, desch);
	

    /* output data to our global data structure */
    if (ok)
      while (subsets--) 
        ok = bufr_parse_out (dds, 0, ndescs - 1, our_callback, 1);
	

    /* close bitstreams and free descriptor array */
    if (dds != (dd*) NULL)
        free (dds);
    bufr_close_descsec_r (desch);
    bufr_close_datasect_r ();


#ifdef VERBOSE
    { int i;  for (i = 0; i < 6; i++) { fprintf (stderr, "section %d length = %d\n", i, bufr_msg.secl[i]);}    }
#endif

    bufr_free_data (&bufr_msg);
    free_descs();
    
    
    /* get data from global */           
    /*fprintf (stderr, "OK: Probando Struct Referencia: Meta de OurData \n");
    radar_data_t* b = &our_data;
    fprintf (stderr, "lon_meta: %f\n", b->meta.radar.lat);
    fprintf (stderr, "lat_meta: %f\n", b->meta.radar.lon);*/
    /*fprintf (stderr, "Extraccion de Datos con Exito.");*/
    
    radar_data_t* b = &our_data;
    return &(b->meta);
}



int* get_data (char* buffile, char* table_dir)
{
    flag_data = 1;
    /*char *table_dir = NULL;*/
    sect_1_t s1;
    bufr_t bufr_msg;    /* structure holding encoded bufr message */
	int ok, desch, ndescs, subsets;
    dd* dds = NULL;

    /* initialize variables */
    memset (&bufr_msg, 0, sizeof (bufr_t));
    memset (&s1, 0, sizeof (sect_1_t));
    memset (&our_data, 0, sizeof (radar_data_t));


    /* read source-file. Therefore allocate memory to hold the complete
       BUFR-message */
    if (!bufr_read_file (&bufr_msg, buffile)) {
        bufr_free_data (&bufr_msg);
        exit (EXIT_FAILURE);    }

    /* decode section 1 */
    if (!bufr_decode_sections01 (&s1, &bufr_msg)) {
        bufr_free_data (&bufr_msg);
        exit (EXIT_FAILURE);    }


    /* read descriptor tables - carga descriptores de ambas tablas en vector de objetos descriptores des[]*/
    if (read_tables (table_dir, s1.vmtab, s1.vltab, s1.subcent, s1.gencent) < 0) {
        bufr_free_data (&bufr_msg);
        free_descs();
        exit (EXIT_FAILURE);    }

    /* decode data descriptor and data-section now */
    /* open bitstreams for section 3 and 4 */
    desch = bufr_open_descsec_r(&bufr_msg, &subsets);    
    ok = (desch >= 0);       
    if (ok) ok = (bufr_open_datasect_r(&bufr_msg) >= 0);
	
	
    /* calculate number of data descriptors  */  
    ndescs = bufr_get_ndescs (&bufr_msg);

    
    /* allocate memory and read data descriptors from bitstream */
    if (ok) ok = bufr_in_descsec (&dds, ndescs, desch);
	

    /* output data to our global data structure */
    if (ok)
      while (subsets--) 
        ok = bufr_parse_out (dds, 0, ndescs - 1, our_callback, 1);
	

    /* close bitstreams and free descriptor array */
    if (dds != (dd*) NULL)
        free (dds);
    bufr_close_descsec_r (desch);
    bufr_close_datasect_r ();
	

#ifdef VERBOSE
    { int i;  for (i = 0; i < 6; i++) { fprintf (stderr, "section %d length = %d\n", i, bufr_msg.secl[i]);}    }
#endif

    bufr_free_data (&bufr_msg);
    free_descs();
    
    /*fprintf (stderr, "Extraccion de Datos con Exito.");*/
    return &sweep_data;   
}




int get_size_data (char* buffile, char* table_dir)
{   
    flag_size = 1;
    /*char *table_dir = NULL;*/
    sect_1_t s1;
    bufr_t bufr_msg;    /* structure holding encoded bufr message */
	int ok, desch, ndescs, subsets;
    dd* dds = NULL;


    /* initialize variables */
    memset (&bufr_msg, 0, sizeof (bufr_t));
    memset (&s1, 0, sizeof (sect_1_t));
    memset (&our_data, 0, sizeof (radar_data_t));


    /* read source-file. Therefore allocate memory to hold the complete
       BUFR-message */
    if (!bufr_read_file (&bufr_msg, buffile)) {
        bufr_free_data (&bufr_msg);
        exit (EXIT_FAILURE);
    }

    /* decode section 1 */
    if (!bufr_decode_sections01 (&s1, &bufr_msg)) {
        bufr_free_data (&bufr_msg);
        exit (EXIT_FAILURE);
    }


    /* read descriptor tables - carga descriptores de ambas tablas en vector de objetos descriptores des[]*/
    if (read_tables (table_dir, s1.vmtab, s1.vltab, s1.subcent, s1.gencent) < 0) {
        bufr_free_data (&bufr_msg);
        free_descs();
        exit (EXIT_FAILURE);
    }

  	
     /* decode data descriptor and data-section now */
    /* open bitstreams for section 3 and 4 */
    desch = bufr_open_descsec_r(&bufr_msg, &subsets);    
    ok = (desch >= 0);       
    if (ok) ok = (bufr_open_datasect_r(&bufr_msg) >= 0);
	
	
    /* calculate number of data descriptors  */  
    ndescs = bufr_get_ndescs (&bufr_msg);
    
    
    /* allocate memory and read data descriptors from bitstream */
    if (ok) ok = bufr_in_descsec (&dds, ndescs, desch);
	

    /* output data to our global data structure */
    /*while (ok && subsets--) */
    if (ok)
      while (subsets--) 
        ok = bufr_parse_out (dds, 0, ndescs - 1, our_callback, 1);
	
	
    /* close bitstreams and free descriptor array */
    if (dds != (dd*) NULL)
        free (dds);
    bufr_close_descsec_r (desch);
    bufr_close_datasect_r ();
	

#ifdef VERBOSE
    { int i;  for (i = 0; i < 6; i++) { fprintf (stderr, "section %d length = %d\n", i, bufr_msg.secl[i]);}    }
#endif

    bufr_free_data (&bufr_msg);
    free_descs();
    
    /*fprintf (stderr, "Extraccion de Datos con Exito.");*/
    return size_data;   
}



int* test9 (char* buffile, char* destfile)
{
	fprintf (stderr, "Iniciando Conversion\n");
    char *table_dir = NULL;
    
    char *sect1_file = "section.9.out";
    char imgfile[200];  /* filename of uncompressed image */
    sect_1_t s1;
    bufr_t bufr_msg;    /* structure holding encoded bufr message */


    /* initialize variables */
    memset (&bufr_msg, 0, sizeof (bufr_t));
    memset (&s1, 0, sizeof (sect_1_t));
    memset (&our_data, 0, sizeof (radar_data_t));


    /* read source-file. Therefore allocate memory to hold the complete
       BUFR-message */
    if (!bufr_read_file (&bufr_msg, buffile)) {
        bufr_free_data (&bufr_msg);
        exit (EXIT_FAILURE);
    }

    /* decode section 1 */
    if (!bufr_decode_sections01 (&s1, &bufr_msg)) {
        bufr_free_data (&bufr_msg);
        exit (EXIT_FAILURE);
    }

	fprintf (stderr, "S1.vmtab: %d\n",s1.vmtab );
	fprintf (stderr, "S1.vltab: %d\n",s1.vltab );
	fprintf (stderr, "S1.subcent: %d\n",s1.subcent );

    /* Write section 1 to ASCII file */
    if (!bufr_sect_1_to_file (&s1, sect1_file)) {
        bufr_free_data (&bufr_msg);
        exit (EXIT_FAILURE);
    }

    /* read descriptor tables - carga descriptores de ambas tablas en vector de objetos descriptores des[]*/
    if (read_tables (table_dir, s1.vmtab, s1.vltab, s1.subcent, s1.gencent) < 0) {
        bufr_free_data (&bufr_msg);
        free_descs();
        exit (EXIT_FAILURE);
    }

    /* decode data descriptor- and data-section now */
    if (!bufr_data_to_file (destfile, imgfile, &bufr_msg)) {
        fprintf (stderr, "unable to decode BUFR-message !\n");
        bufr_free_data (&bufr_msg);
        free_descs();
        exit (EXIT_FAILURE);
    } 
    
	
    int ok, desch, ndescs, subsets;
    dd* dds = NULL;

    /* decode data descriptor and data-section now */
    /* open bitstreams for section 3 and 4 */
    desch = bufr_open_descsec_r(&bufr_msg, &subsets);
    
    fprintf (stderr, "Numero de desch: %d\n",desch);
    fprintf (stderr, "Numero de Subsets: %d\n",subsets);
       
    ok = (desch >= 0);       
    if (ok) ok = (bufr_open_datasect_r(&bufr_msg) >= 0);
	
	
    /* calculate number of data descriptors  */  
    ndescs = bufr_get_ndescs (&bufr_msg);
    fprintf (stderr, "Numero de Data Descriptors: %d\n",ndescs);
    
    /* allocate memory and read data descriptors from bitstream */
    if (ok) ok = bufr_in_descsec (&dds, ndescs, desch);
	

    /* output data to our global data structure */
    /*while (ok && subsets--) */
    if (ok)
      while (subsets--) 
        ok = bufr_parse_out (dds, 0, ndescs - 1, our_callback, 1);
	

    /* close bitstreams and free descriptor array */
    if (dds != (dd*) NULL)
        free (dds);
    bufr_close_descsec_r (desch);
    bufr_close_datasect_r ();
	

#ifdef VERBOSE
    { int i;  for (i = 0; i < 6; i++) { fprintf (stderr, "section %d length = %d\n", i, bufr_msg.secl[i]);}    }
#endif

    bufr_free_data (&bufr_msg);
    free_descs();
    /*exit (EXIT_SUCCESS);*/
    
    
    /* get data from global 
    radar_data_t* b = &our_data;
    return &(b->meta.radar);*/
           
    fprintf (stderr, "OK: Probando Struct Referencia: Meta de OurData \n");
    radar_data_t* b = &our_data;
    fprintf (stderr, "lon_meta: %f\n", b->meta.radar.lat);
    fprintf (stderr, "lat_meta: %f\n", b->meta.radar.lon);
    
    fprintf (stderr, "OK: Fin de la Funcion, RETORNANDO... \n");
    
    return &sweep_data;
}



static int our_callback (varfl val, int ind) {

    radar_data_t* b = &our_data;   /* our global data structure */
    bufrval_t* v;                  /* array of data values */
    
    int i = 0;
    int nv, nr, nc;
    dd* d;

    /* do nothing if data modifictaon descriptor or replication descriptor */
    if (ind == _desc_special) {
		fprintf (stderr, "Replicator Factor o data modification descriptor\n");
		return 1;}
		
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

        
        /* Seccion Titulo */
        else if   (bufr_check_fxy (d, 3,21,204)) {   

            /* decode sequence to global array */
            bufr_parse_out (des[ind]->seq->del, 0, des[ind]->seq->nel - 1,
                            bufr_val_to_global, 0);
            /* get our data from the array */            
			/*i++;
			fprintf (stderr, "\nSección TItulo\n");
			fprintf (stderr, "Radar: %c\n",(char) v->vals[i++]);
			fprintf (stderr, "Radar: %c\n",(char) v->vals[i++]);
			fprintf (stderr, "Radar: %c\n",(char) v->vals[i++]);
			fprintf (stderr, "Radar: %c\n",(char) v->vals[i++]);*/
        }
                
        
        /* Seccion Volumen */
        else if   (bufr_check_fxy (d, 3,1,31)) {   

            /* decode sequence to global array */
            bufr_parse_out (des[ind]->seq->del, 0, des[ind]->seq->nel - 1,
                            bufr_val_to_global, 0);
			
			vv = v->vals;
            i = 0;
            
            /*fprintf (stderr, "\nSección Volumen\n");*/
            /*3  1  31       0.0000000   0  1   1 WMO block number
                 0.0000000   0  1   2 WMO station number
                 0.0000000   0  2   1 Type of station
              2017.0000000   0  4   1 Year
                 3.0000000   0  4   2 Month
                28.0000000   0  4   3 Day
                 4.0000000   0  4   4 Hour
                43.0000000   0  4   5 Minute
               -31.4413300   0  5   1 Latitude (high accuracy)
               -64.1919200   0  6   1 Longitude (high accuracy)
               484.0000000   0  7   1 Height of station*/
            
            /* get our data from the array */
            b->wmoblock = 	(int) vv[i++];
            b->wmostat = 	(int) vv[i++];
            i++;
            b->meta.year = (int) vv[i++];       	/* Date */
            b->meta.month = (int) vv[i++];
            b->meta.day = (int) vv[i++];
            b->meta.hour = (int) vv[i++];       	/* Time */
            b->meta.min = (int) vv[i++];
            b->meta.radar.lat = vv[i++];        	/* Latitude of radar */
            b->meta.radar.lon = vv[i++];        	/* Longitude of radar */
            b->meta.radar_height = vv[i++];         /* Altura */           
        }
        
        
        
		/* Seccion Barridos */
        else if   (bufr_check_fxy (d, 3,21,203)) {   

            /* decode sequence to global array */
            bufr_parse_out (des[ind]->seq->del, 0, des[ind]->seq->nel - 1,
                            bufr_val_to_global, 0);
			



			vv = v->vals;
			
			if (flag_data==1) { 
				/* fprintf (stderr, "Indice: 0, Valor: %d\n",(int) vv[0]);	*/ 	/*15*/
				/* fprintf (stderr, "Indice: 1, Valor: %d\n",(int) vv[1]); 	*/	/*2017*/
				/* fprintf (stderr, "Indice: 23, Valor: %d\n",(int) vv[23]);*/	/*9*/
				/* fprintf (stderr, "Indice: 24, Valor: %d\n",(int) vv[24]);*/	/*65534*/
				
				int u=24;
				int nbarridos = (int)vv[0];
				int multi_pri = (int)vv[23];
				int j;
				int b;
				for (b=0; b<nbarridos; b++) {

					for (j=0; j<multi_pri; j++){
						u=u+(int)vv[u]+1;
						/*fprintf (stderr, "Indice: %d, Valor: %d\n",u, (int) vv[u]);*/
					}

					if (b!=nbarridos-1) {
					u = u+22;
					multi_pri = (int)vv[u];
					/*fprintf (stderr, "\nMulti Pri: %d, Valor: %d\n",u, (int) vv[u]);*/
					u++;
					/*fprintf (stderr, "Multi Sec1: %d, Valor: %d\n",u, (int) vv[u]);*/
					}
				}
				
				
				i = 0;
				int lim = u;
				for (j=0; j<lim; j++){
					sweep_data[j] = (int) vv[j]; 
					}
			}
					


			if (flag_size==1) { 
				int u=24;
				int nbarridos = (int)vv[0];
				int multi_pri = (int)vv[23];
				int j;
				int b;
				for (b=0; b<nbarridos; b++) {

					for (j=0; j<multi_pri; j++){
						u=u+(int)vv[u]+1;
					}

					if (b!=nbarridos-1) {
					u = u+22;
					multi_pri = (int)vv[u];
					u++;
					}
				}
				size_data = u;
				
			}


/* 3 21 203      15.0000000   0 31   1 Delayed descriptor replication factor
             2017.0000000   0  4   1 Year
                3.0000000   0  4   2 Month
               28.0000000   0  4   3 Day
                4.0000000   0  4   4 Hour
               43.0000000   0  4   5 Minute
               27.0000000   0  4   6 Second
             2017.0000000   0  4   1 Year
                3.0000000   0  4   2 Month
               28.0000000   0  4   3 Day
                4.0000000   0  4   4 Hour
               43.0000000   0  4   5 Minute
               52.0000000   0  4   6 Second
               90.0000000   0 30 196 Type of product
                0.5300000   0  2 135 Antenna elevation
              787.0000000   0 30 194 Number of bins along the radial
              300.0000000   0 21 201 Range-bin size
             1740.0000000   0 21 203 Range-bin offset
              360.0000000   0 30 195 Number of azimuths
               81.0000000   0  2 134 Antenna beam azimuth
                1.0000000   0 31   1 Delayed descriptor replication factor
              230.0000000   0 30 196 Type of product
                0.0000000   0 30 197 Compression method
                9.0000000   0 31   2 Extended delayed descriptor replication factor
            65534.0000000   0 31   2 Extended delayed descriptor replication factor
*/

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



void bufr_decoding_sample (bufr_t* msg, radar_data_t* data) {

	char *table_dir = "/home/jsaffe/Descargas/OPERA/bufr_3.2/";
	
    sect_1_t s1;
    int ok, desch, ndescs, subsets;
    dd* dds = NULL;

    /* initialize variables */
    memset (&s1, 0, sizeof (sect_1_t));

    /* decode section 1 */
    ok = bufr_decode_sections01 (&s1, msg);

    /* Write section 1 to ASCII file */
    bufr_sect_1_to_file (&s1, "section.1.out");

    /* read descriptor tables */
    if (ok) ok = (read_tables (table_dir, s1.vmtab, s1.vltab, s1.subcent, 
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
