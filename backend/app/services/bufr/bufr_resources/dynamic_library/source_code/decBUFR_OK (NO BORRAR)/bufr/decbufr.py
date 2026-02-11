#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:10:26 2017

@author: jsaffe
"""

from ctypes import POINTER, Structure, c_char_p, c_double, c_int, cdll

import numpy as np


class point_t(Structure):
    _fields_ = [
        ("lat",  c_double),
        ("lon",  c_double)]

class meta_t(Structure):
    _fields_ = [
        ("year",  c_int),
        ("month", c_int),
        ("day",   c_int),
        ("hour",  c_int),
        ("min",   c_int),
        ("radar", point_t),
        ("radar_height", c_double)]


bufr_filename= '/media/jsaffe/DATOS/BBDD_RADARES/L2/BUFR/RMA1/2017/03/28/05/3357/RMA1_0117_01_TV_20170328T053357Z.BUFR'
root_tables_bufr = './'
#bufr_name= '/home/jsaffe/Descargas/OPERA/bufr_3.2/RMA1_0117_01_TH_20170328T044327Z.BUFR'
#==============================================================================
# DECBUFR get_meta_data
#==============================================================================
libtest=cdll.LoadLibrary("./libdecbufr.so") 
get_meta_data = libtest.get_meta_data
get_meta_data.argtypes  = [c_char_p, c_char_p]
get_meta_data.restype   = POINTER (meta_t)
volumen_metadata = get_meta_data(bufr_filename, root_tables_bufr)

#print '\nMETA DATA:'
#print 'Año', volumen_metadata.contents.year
#print 'Mes', volumen_metadata.contents.month
#print 'Dia', volumen_metadata.contents.day
#print '\n'
#print volumen_metadata.contents.hour
#print volumen_metadata.contents.min
#print volumen_metadata.contents.radar.lat
#print volumen_metadata.contents.radar.lon
#print volumen_metadata.contents.radar_height

#==============================================================================
# DECBUFR get_size_data
#==============================================================================
get_data_part= libtest.get_size_data
get_data_part.argtypes = [c_char_p, c_char_p]
get_data_part.restype = c_int
size_data=get_data_part(bufr_filename, root_tables_bufr)
size_data=size_data
#print '\nSIZE DATA: ', size_data
#print '\n'
#

#==============================================================================
# DECBUFR get_data_part
#==============================================================================
get_data_part= libtest.get_data
get_data_part.argtypes = [c_char_p, c_char_p]
array=c_int*size_data
get_data_part.restype = POINTER(array)
vol_data=get_data_part(bufr_filename, root_tables_bufr)

vol=list(vol_data.contents)
vol=np.asarray(vol)
del vol_data
#print '\nTAMAÑO VECTOR RECUPERADO: ',np.size(vol)
#print 'Numero Repeticiones: ', vol_data.contents[0]




#==============================================================================
# Parser
#==============================================================================

#* 3 21 203      15.0000000   0 31   1 Delayed descriptor replication factor
#             2017.0000000   0  4   1 Year
#                3.0000000   0  4   2 Month
#               28.0000000   0  4   3 Day
#                4.0000000   0  4   4 Hour
#               43.0000000   0  4   5 Minute
#               27.0000000   0  4   6 Second
#             2017.0000000   0  4   1 Year
#                3.0000000   0  4   2 Month
#               28.0000000   0  4   3 Day
#                4.0000000   0  4   4 Hour
#               43.0000000   0  4   5 Minute
#               52.0000000   0  4   6 Second
#               90.0000000   0 30 196 Type of product
#                0.5300000   0  2 135 Antenna elevation
#              787.0000000   0 30 194 Number of bins along the radial
#              300.0000000   0 21 201 Range-bin size
#             1740.0000000   0 21 203 Range-bin offset
#              360.0000000   0 30 195 Number of azimuths
#               81.0000000   0  2 134 Antenna beam azimuth
#                1.0000000   0 31   1 Delayed descriptor replication factor
#              230.0000000   0 30 196 Type of product
#                0.0000000   0 30 197 Compression method
#                9.0000000   0 31   2 Extended delayed descriptor replication factor
#            65534.0000000   0 31   2 Extended delayed descriptor replication factor
#*/

volumen=[]

nsweeps = vol[0]
u=1
for sweep in range (nsweeps):
    barrido = {}
    barrido['compress_data'] = []
    print 'First ','i:',u,'Valor:', vol[u]
    barrido['Year']=vol[u];  u=u+1;
    barrido['Month']=vol[u]; u=u+1;
    barrido['Day']=vol[u];   u=u+1;
    barrido['Hour']=vol[u];  u=u+1;
    barrido['Minute']=vol[u];u=u+1;
    barrido['Second']=vol[u];u=u+1;
    barrido['Year']=vol[u];  u=u+1;
    barrido['Month']=vol[u]; u=u+1;
    barrido['Day']=vol[u];   u=u+1;
    barrido['Hour']=vol[u];  u=u+1;
    barrido['Minute']=vol[u];u=u+1;
    barrido['Second']=vol[u];u=u+1;
    barrido['Type_of_product']=vol[u];  u=u+1;
    barrido['elevation']=vol[u];        u=u+1;
    barrido['ngates']=vol[u];           u=u+1;
    barrido['range_size']=vol[u];       u=u+1;
    barrido['range_offset']=vol[u];     u=u+1;
    barrido['nrays']=vol[u];            u=u+1;
    barrido['antenna_beam_az']=vol[u];  u=u+2;
    barrido['Type of product']=vol[u];  u=u+2    

    print 'MultiP','i:',u,'Valor:', vol[u]
    multi_pri = vol[u]; u=u+1   
    for i in range(multi_pri):

        print 'MultiS','i:',u,'Valor:', vol[u]
        multi_sec = vol[u]; u=u+1;
        for j in range(multi_sec):
            if (vol[u]==99999):
                barrido['compress_data'].append(255)
            else:
                barrido['compress_data'].append(vol[u])   
            u=u+1
        
#    barrido['data']=np.asarray(barrido['data'], dtype=np.uint8)
    barrido['compress_data'] = bytearray(barrido['compress_data'])
    volumen.append(barrido)


import struct

#==============================================================================
# Descompresion 
#==============================================================================
import zlib

for sweep in range(nsweeps):
    volumen[sweep]['data']=[]
    data_buf = buffer( volumen[sweep]['compress_data'] )
    dec_data=zlib.decompress(data_buf)
    dec_data_bytes=bytearray(dec_data)
    
    j=0
    for i in range (np.size(dec_data_bytes)/8):
        volumen[sweep]['data'].append(struct.unpack('d', dec_data_bytes[j:j+8])[0])
        j=j+8
       
       
#Test       
#data = volumen[0]['data']
#print np.size(data)

#data_buf = buffer( volumen[0]['compress_data'] )
#dec_data=zlib.decompress(data_buf)
#dec_data_bytes=bytearray(dec_data)
#np.size(dec_data_bytes)
#print struct.unpack('d', dec_data_bytes[0:8])

#==============================================================================
# Enmascarado y Redimensionamiento de Informacion
#==============================================================================
for sweep in range(nsweeps):
    volumen[sweep]['data']= np.ma.masked_equal(volumen[sweep]['data'], -1.797693134862315708e+308)
    volumen[sweep]['data']= np.reshape(volumen[sweep]['data'], (volumen[sweep]['nrays'] , volumen[sweep]['ngates']))

#data = volumen[0]['data']


