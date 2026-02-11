# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:53:17 2015
@author: jsaffe
"""

import os
from ctypes import POINTER, Structure, c_char_p, c_double, c_int, cdll
from datetime import datetime

import numpy as np
import numpy.ma as ma
import pyart
import scipy.io
from grc import global_parameters as cf


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



"""####################################################################################
# VERIFICAR:
# a) Chequear en estrategia, si dentro de un mismo volumen puede haber barridos con distinto número de gates del mismo producto (si se puede agregar a lista de por HACER).
# b) Carga de az inicial en los datos (poner 0?)
####################################################################################"""
#------------------------------------------------------------------------------------------------------------------------
# ERROR TIEMPOS:
# Los .m estan bien y el parseo pareciera estar bien, pero hay veces que toma bien los campos de los .m y hay veces que no.
# Es como si hubiese algun error en la libreria que lee los archivos de matlab o quedara algo en memoria que a la hora de la conversión 
# esta generando el error.
#------------------------------------------------------------------------------------------------------------------------


def import_matlab_matrix(path, file_name, debug=False):      
    vol_m = scipy.io.loadmat(path+file_name)   #Lectura archivo matlab
    
    vol={}
    vol['info_volumen'] =vol_m['vol_info'][0][1]            #Cargamos la Información del Volumen
    vol['info_barridos']=vol_m['vol_info'][0][2]            #Cargamos la Información de los Barridos
    vol['info_barridos']=np.squeeze(vol['info_barridos'])   #Elimina primera dimension
    vol['matriz_datos'] =vol_m['vol_info'][0][0]            #Cargamos la Matriz de Datos de nsweeps_nrayos_x_n_gates
    
    #......................................................................................
    # DATOS VOLUMEN: 
    #   Reordenamos los datos en función del formato esperado por PyART. 
    #   Formato esperado: nrayostotales_x_n_gates
    #......................................................................................  
    #--------------------------------------------------------------------------------
    # Estrategia 9005
    #--------------------------------------------------------------------------------
    #   Parche momentaneo para Estrategia 9005. El 4to volumen tiene los 2 ultimos 
    #   barridos con dimensiones distintas a las primeras. Para procesar esto se debe 
    #   actualizar la version completa del conversor, no solo esta parte
    #---------------------------------------------------------------------------------    
    if (file_name.split('_')[1] == '9005'):
        if (debug):
            print 'Estrategia 9005'        
        
        num_volumen = file_name.split('_')[2]    
        if num_volumen != '04':  
            vol['data'] = vol['matriz_datos'][0][0]
            for i in range(1,(vol['matriz_datos'][0].shape[0])):
                vol['data']=np.concatenate((vol['data'], vol['matriz_datos'][0][i]), axis=0)    
        elif num_volumen == '04':
            vol['data']=np.concatenate((vol['matriz_datos'][0][5],vol['matriz_datos'][0][6]), axis=0)    #copiio solo los ultimos 2 barridos (no son de calibración, tienen info util)
            
            vol['info_barridos'] = [vol['info_barridos'][5],vol['info_barridos'][6]]
            vol['info_volumen'][0][0][9][0][0]=2 #nseeps

    #--------------------------------------------------------------------------------
    # Estrategias Generales
    #--------------------------------------------------------------------------------
    else:
        vol['data'] = vol['matriz_datos'][0][0]
        for i in range(1,(vol['matriz_datos'][0].shape[0])):
           vol['data']=np.concatenate((vol['data'], vol['matriz_datos'][0][i]), axis=0)    
    #---------------------------------------------------------------------------------    


    #Enmascara los Datos en función de los valores inválidos (NaN, INF, etc)
    vol['data']=ma.masked_invalid(vol['data'])
 
    #........................................................................
    #INFO VOLUMEN
    #........................................................................
    
    #........................................................................
    #Info de Instrumento y Estrategia
    #........................................................................
    vol['info']={}
    vol['info']['estrategia']={}
    vol['info']['metadata']={}
    
    
    vol['info']['nombre_radar']=file_name.split('_')[0]
    vol['info']['estrategia']['nombre']=file_name.split('_')[1]
    vol['info']['estrategia']['volume_number']=file_name.split('_')[2]
    vol['info']['tipo_producto']=file_name.split('_')[3]    
    
    #........................................................................
    #Carga de Info General del Volumen
    #........................................................................
    vol['info']['ano_vol']=vol['info_volumen'][0][0][0][0][0]
    vol['info']['mes_vol']=vol['info_volumen'][0][0][1][0][0]
    vol['info']['dia_vol']=vol['info_volumen'][0][0][2][0][0]
    vol['info']['hora_vol']=vol['info_volumen'][0][0][3][0][0]
    vol['info']['min_vol']=vol['info_volumen'][0][0][4][0][0]
    #vol['info']['seg_vol']=vol['info_volumen'][0][0][5][0][0]
    vol['info']['lat']=vol['info_volumen'][0][0][6][0][0]
    vol['info']['lon']=vol['info_volumen'][0][0][7][0][0]
    vol['info']['altura']=vol['info_volumen'][0][0][8][0][0]
    vol['info']['nsweeps']=vol['info_volumen'][0][0][9][0][0]
    
    #........................................................................
    #Carga de Info de Barridos
    #........................................................................
    nsweeps=vol['info']['nsweeps']
    vol['info']['ano_sweep_ini']=np.zeros(nsweeps); 
    vol['info']['mes_sweep_ini']=np.zeros(nsweeps); 
    vol['info']['dia_sweep_ini']=np.zeros(nsweeps); 
    vol['info']['hora_sweep_ini']=np.zeros(nsweeps); 
    vol['info']['min_sweep_ini']=np.zeros(nsweeps);
    vol['info']['seg_sweep_ini']=np.zeros(nsweeps);
    vol['info']['ano_sweep']=np.zeros(nsweeps);
    vol['info']['mes_sweep']=np.zeros(nsweeps); 
    vol['info']['dia_sweep']=np.zeros(nsweeps); 
    vol['info']['hora_sweep']=np.zeros(nsweeps); 
    vol['info']['min_sweep']=np.zeros(nsweeps); 
    vol['info']['seg_sweep']=np.zeros(nsweeps)
    vol['info']['elevaciones']=np.zeros(nsweeps)
    vol['info']['ngates']=np.zeros(nsweeps); 
    vol['info']['gate_size']=np.zeros(nsweeps); 
    vol['info']['gate_offset']=np.zeros(nsweeps)
    vol['info']['nrayos']=np.zeros(nsweeps); 
    vol['info']['rayo_inicial']=np.zeros(nsweeps)
    
    for sweep in range (0,nsweeps):
        vol['info']['ano_sweep_ini'] [sweep]=int(vol['info_barridos'][sweep][0][0][0])
        vol['info']['mes_sweep_ini'] [sweep]=int(vol['info_barridos'][sweep][1][0][0])
        vol['info']['dia_sweep_ini'] [sweep]=int(vol['info_barridos'][sweep][2][0][0])
        vol['info']['hora_sweep_ini'][sweep]=int(vol['info_barridos'][sweep][3][0][0])
        vol['info']['min_sweep_ini'] [sweep]=int(vol['info_barridos'][sweep][4][0][0])
        vol['info']['seg_sweep_ini'] [sweep]=int(vol['info_barridos'][sweep][5][0][0])
    
        vol['info']['ano_sweep'] [sweep]=int(vol['info_barridos'][sweep][6][0][0])
        vol['info']['mes_sweep'] [sweep]=int(vol['info_barridos'][sweep][7][0][0])
        vol['info']['dia_sweep'] [sweep]=int(vol['info_barridos'][sweep][8][0][0])
        vol['info']['hora_sweep'][sweep]=int(vol['info_barridos'][sweep][9][0][0])
        vol['info']['min_sweep'] [sweep]=int(vol['info_barridos'][sweep][10][0][0])
        vol['info']['seg_sweep'] [sweep]=int(vol['info_barridos'][sweep][11][0][0])
    
        vol['info']['elevaciones'][sweep]=vol['info_barridos'][sweep][12][0][0]
        vol['info']['ngates'][sweep]=int(vol['info_barridos'][sweep][13][0][0])
        vol['info']['gate_size'][sweep]=vol['info_barridos'][sweep][14][0][0]
        vol['info']['gate_offset'][sweep]=vol['info_barridos'][sweep][15][0][0]
        vol['info']['nrayos'][sweep]=vol['info_barridos'][sweep][16][0][0]
        vol['info']['rayo_inicial'][sweep]=vol['info_barridos'][sweep][17][0][0]
        
    #........................................................................
    #Info para Metadata
    #........................................................................
    vol['info']['metadata']['comment']             ='-'
    vol['info']['metadata']['instrument_type']     ='Radar'
    vol['info']['metadata']['site_name']           ='-'
    vol['info']['metadata']['Sub_conventions']     ='-'
    vol['info']['metadata']['references']          ='-'
    vol['info']['metadata']['volume_number']       = vol['info']['estrategia']['volume_number']
    vol['info']['metadata']['scan_id']             = vol['info']['estrategia']['nombre']
    vol['info']['metadata']['title']               ='-'
    vol['info']['metadata']['source']              ='-'
    vol['info']['metadata']['version']             ='-'
    vol['info']['metadata']['instrument_name']     = vol['info']['nombre_radar']
    vol['info']['metadata']['ray_times_increase']  ='-'
    vol['info']['metadata']['platform_is_mobile']  ='false'
    vol['info']['metadata']['driver']              ='-'
    vol['info']['metadata']['institution']         ='SiNaRaMe'
    vol['info']['metadata']['n_gates_vary']        ='-'
    vol['info']['metadata']['primary_axis']        ='-'
    vol['info']['metadata']['created']             ='Fecha:'+str(int(vol['info']['dia_sweep'][0]))+'/'+str(int(vol['info']['mes_sweep'][0]))+'/'+str(int(vol['info']['ano_sweep'][0]))+' Hora:'+str(int(vol['info']['hora_sweep'][0]))+':'+str(int(vol['info']['min_sweep'][0]))+':'+str(int(vol['info']['seg_sweep'][0]))
    vol['info']['metadata']['scan_name']           ='-'
    vol['info']['metadata']['author']              ='Grupo Radar Cordoba (GRC) - Extractor/Conversor de Datos de Radar '
    vol['info']['metadata']['Conventions']         ='-'
    vol['info']['metadata']['platform_type']       ='Base Fija'
    vol['info']['metadata']['history']             ='-'    
    
    #........................................................................
    #Limpiamos el diccionario
    #........................................................................
    del vol['info_barridos']    
    del vol['info_volumen']    
    del vol['matriz_datos']    
    
    return vol

       



def matlab_matrix_to_PyARTobject(path, save_file=False, path_out=0, debug=False):
    lstDir = os.walk(path)   #os.walk() Lista directorios y ficheros #Lista con todos los ficheros del directorio:
    
    volumenes=[]
    i=0
    files_in_dir=0
    
    for root, dirs, files in lstDir:        
        for fichero in files:
            
            ext_fichero=fichero.split(".")[np.size(fichero.split("."))-1] #chequeamos si el archivo actual es .mat
            if ext_fichero == "mat":
                
                if (debug):
                    print fichero #test
                files_in_dir=1
                nom_fichero=fichero
                
                #---------------------------------------------------------------------            
                #CARGA LOS DATOS DEL VOLUMEN
                #---------------------------------------------------------------------
                vol={}
                vol=import_matlab_matrix(path+'/', fichero, debug=debug)
                
                #---------------------------------------------------------------------            
                #CHEQUEA COMPATIBILIDAD DEL VOLUMEN_PRODUCTO (segun lo implementado actualmente)
                #---------------------------------------------------------------------
                for sweep in range (1, vol['info']['nsweeps']):            
                    #Los nrayos de todos los sweeps deben ser iguales
                    if vol['info']['nrayos'][0] != vol['info']['nrayos'][sweep]:
                        print "ERROR: Volumen no soportado, número de rayos distintos entre sweeps"                    
                        #exit
                
                    #Los ngates de todos los sweeps deben ser iguales    
                    if vol['info']['ngates'][0] != vol['info']['ngates'][sweep]:
                        print "ERROR: Volumen no soportado, número de gates distintos entre sweeps"                    
                        #exit
                
                    #Los gate_offset de todos los sweeps deben ser iguales    
                    if vol['info']['gate_offset'][0] != vol['info']['gate_offset'][sweep]:
                        print "ERROR: Volumen no soportado, valores de gate_offset distintos entre sweeps"                    
                        #exit
    
                    #Los gate_offset de todos los sweeps deben ser iguales    
                    if vol['info']['gate_size'][0] != vol['info']['gate_size'][sweep]:
                        print "ERROR: Volumen no soportado, valores de gate_size  distintos entre sweeps"                    
                        #exit
                    
            
                #--------------------------------------------------------------------- 
                #Agregamos Volumen Chequeado a Lista Volumenes
                #--------------------------------------------------------------------- 
                #print 'LECTURA '+vol['info']['tipo_producto']+': OK'
                volumenes.append(vol)
        
                #---------------------------------------------------------------------            
                #CHEQUEA que todos los VOLUMENES (PRODUCTOS) CARGADOS PERTENEZCAN a la mimas fecha y hora.
                #---------------------------------------------------------------------
                if i==0:            
                    fecha_vol=fichero.split('_')[4]
                    i=1
                
                else:
                    if fecha_vol != fichero.split('_')[4] :
                        print "ERROR: Los productos en el directorio no pertenecen al mismo Volumen"                    
                        exit
    

    if files_in_dir:                
        #---------------------------------------------------------------------            
        #DETECTA PRODUCTO DE REFERENCIA (aquel que tiene el range mas lejano)
        #---------------------------------------------------------------------
        rango_maximo=np.zeros(len(volumenes))
        for nvolumen in range (0,len(volumenes)):
            rango_maximo[nvolumen]=volumenes[nvolumen]['info']['gate_offset'][0]+(volumenes[nvolumen]['info']['gate_size'][0]*volumenes[nvolumen]['info']['ngates'][0])      
        nreferencia=np.argmax(rango_maximo) #retorna el índice del máximo valor en el array.
                
        #---------------------------------------------------------------------
        #CARGA DE DATOS e INFO del Volumen de Referencia
        #---------------------------------------------------------------------           
        vol=volumenes[nreferencia]    
        #print "\n \n \n VOL(ref)_"+vol['info']['tipo_producto']+"\n nrayos:"+str(vol['info']['nrayos'])+"\n ngates: "+str(vol['info']['ngates'])+"\n gate_size: "+str(vol['info']['gate_size'])+"\n gate_offset: "+str(vol['info']['gate_offset'])                
        #print "\n Hora:",vol['info']['hora_sweep'],"\n min:",vol['info']['min_sweep'],"\n seg:",vol['info']['seg_sweep']
        
        #......................................................
        #Creación del Objeto PyART
        #LIMITACION: esta forma de crear los archivos impone la limitación 
        #que todos los barridos debe tener el mismo número de gates y rayos.            
        #......................................................
        radar = pyart.testing.make_empty_ppi_radar(vol['info']['ngates'][0], vol['info']['nrayos'][0], vol['info']['nsweeps'])
        
        #......................................................
        #Carga de Datos del Producto de Referencia en Objeto-PyART
        #......................................................
        radar.add_field(vol['info']['tipo_producto'], vol, replace_existing=True)
                      
        #......................................................
        #Carga de Campos de Informacion General en Objeto-PyART
        #......................................................
        #----------------------------------------------------------------- 
        #Altura del Radar
        #-----------------------------------------------------------------
        radar.altitude['data']=np.ndarray(1)
        radar.altitude['data'][0]=vol['info']['altura']
        radar.altitude['units']='metros'
        radar.altitude['long_name']='altitud'
        radar.altitude['possitive:']='arriba'
        radar.altitude['_fillValue']=-9999.0

#2      Chequear que esta sea la estructura para este campo, lo copie de altitude   
#2      Chequear si altura de RMAs, incluyen las torres. Sino cargar tabla con altura de torres y restar a altitud esa altura y ponersela a esta
#        radar.altitude_agl['data']=np.ndarray(1)
#        radar.altitude_agl['data'][0]=0
#        radar.altitude_agl['units']='metros'
#        radar.altitude_agl['long_name']='altitud_sobre_nivel_del_suelo'
#        radar.altitude_agl['possitive:']='arriba'
#        radar.altitude_agl['_fillValue']=-9999.0


        #-----------------------------------------------------------------
        #Azimuth / Elevación / Fixed_angle / 
        #-----------------------------------------------------------------
        z=0
        for i in range (0, vol['info']['nsweeps']):
            
            for j in range (0, int(vol['info']['nrayos'][i])):
                radar.azimuth['data'][z]=j
                radar.elevation['data'][z]=vol['info']['elevaciones'][i]
                z=z+1
            
            radar.fixed_angle['data'][i]=vol['info']['elevaciones'][i]
        
        #-----------------------------------------------------------------
        #Coordenadas Geográficas
        #-----------------------------------------------------------------
        radar.latitude['data']=np.ndarray(1)
        radar.latitude['data'][0]=vol['info']['lat']
        radar.latitude['units']='grados'
        radar.latitude['long_name']='latitud'
        radar.latitude['_fillValue']=-9999.0
        
        radar.longitude['data']=np.ndarray(1)
        radar.longitude['data'][0]=vol['info']['lon']
        radar.longitude['units']='grados'
        radar.longitude['long_name']='longitud'
        radar.longitude['_fillValue']=-9999.0
        
        #-----------------------------------------------------------------
        #Rango
        #-----------------------------------------------------------------
#!      #REVISAR otros archivos de radar si comienzan en 0 o en offset
        
        radar.range['data'][0]=vol['info']['gate_offset'][0] #inicia en el gate_offset
        for i in range (1, int(vol['info']['ngates'][0])):
            radar.range['data'][i]= radar.range['data'][i-1]+vol['info']['gate_size'][0]
        
        radar.range['meters_between_gates']=vol['info']['gate_size'][0]
        radar.range['meters_to_center_of_first_gate']=vol['info']['gate_offset'][0]
        
        
        
        #-----------------------------------------------------------------
        #MetaData
        #-----------------------------------------------------------------
        radar.metadata['comment']=vol['info']['metadata']['comment']                        #Nada
        radar.metadata['instrument_type']=vol['info']['metadata']['instrument_type']        #'Radar'
#2
        radar.metadata['site_name']=vol['info']['metadata']['site_name']                    #Nada       #PROX: cargar listado de sitios 
        radar.metadata['Sub_conventions']=vol['info']['metadata']['Sub_conventions']        #Nada
        radar.metadata['references']=vol['info']['metadata']['references']                  #Nada
        radar.metadata['volume_number']=vol['info']['metadata']['volume_number']            #Volumen dentro de Estrategia
        radar.metadata['scan_id']=vol['info']['metadata']['scan_id']                        #Estrategia
        radar.metadata['title']=vol['info']['metadata']['title']                            #Nada
        radar.metadata['source']=vol['info']['metadata']['source']                          #Nada
        radar.metadata['version']=vol['info']['metadata']['version']                        #Nada
        radar.metadata['instrument_name']=vol['info']['metadata']['instrument_name']        #Nombre Radar
#2
        radar.metadata['ray_times_increase']=vol['info']['metadata']['ray_times_increase']  #Nada       #PROX: Podria decir:'Variable de Barrido a Barrido' o leerla de estrategia y cargar vector con info
        radar.metadata['platform_is_mobile']=vol['info']['metadata']['platform_is_mobile']  #False  
        radar.metadata['driver']=vol['info']['metadata']['driver']                          #Nada
        radar.metadata['institution']=vol['info']['metadata']['institution']                #'SiNaRaMe'
        radar.metadata['n_gates_vary']=vol['info']['metadata']['n_gates_vary']              #'False'            
        radar.metadata['primary_axis']=vol['info']['metadata']['primary_axis']              #Nada
#!
        radar.metadata['created']=vol['info']['metadata']['created']                        #Fecha      #Chequear pq es incongruente la fecha
#2
        radar.metadata['scan_name']=vol['info']['metadata']['scan_name']                    #Nada       #PROX: leer de estrategia el tipo de procesamiento
        radar.metadata['author']=vol['info']['metadata']['author']                          #'Grupo Radar Cordoba (GRaC) - Extractor/Conversor de Datos de Radar' 
        radar.metadata['Conventions']=vol['info']['metadata']['Conventions']                #Nada
        radar.metadata['platform_type']=vol['info']['metadata']['platform_type']            #'Base Fija'
        radar.metadata['history']=vol['info']['metadata']['history']                        #Nada
        
        #*************************************************************************************************
        #TIEMPO
        #*************************************************************************************************
        radar.time['comment']='tiempos relativos al tiempo de inicio del primer barrido del volumen'
        radar.time['long_name']='tiempo en segundos desde inicio del primer barrido del volumen'
        radar.time['standard_name']='tiempo'
        radar.time['units']='seconds since '+str(int(vol['info']['ano_sweep'][0]))+'-'+str(int(vol['info']['mes_sweep'][0]))+'-'+str(int(vol['info']['dia_sweep'][0]))+'T'+str(int(vol['info']['hora_sweep'][0]))+':'+str(int(vol['info']['min_sweep'][0]))+':'+str(int(vol['info']['seg_sweep'][0]))+'Z'
        
        
        #--------------------------------------------------------------------------------
        #Tiempos Iniciales de Barridos (respecto al Tiempo de Inicio del Primer Barrido)
        #--------------------------------------------------------------------------------
        radar.time['tiempo_inicial_sweep']=np.zeros(vol['info']['nsweeps'])
        radar.time['tiempo_inicial_sweep_0']=np.zeros(vol['info']['nsweeps'])
        t_ref = datetime(int(vol['info']['ano_sweep'][0]), int(vol['info']['mes_sweep'][0]), int(vol['info']['dia_sweep'][0]), int(vol['info']['hora_sweep'][0]), int(vol['info']['min_sweep'][0]), int(vol['info']['seg_sweep'][0]))   #Fecha y Hora Inicial del Barrido 0. Lo hacemos referencia, todo el resto estarán referenciados por este.        
        
        for sweep in range (0, vol['info']['nsweeps']):
            fecha1 = datetime(int(vol['info']['ano_sweep'][sweep]), int(vol['info']['mes_sweep'][sweep]), int(vol['info']['dia_sweep'][sweep]), int(vol['info']['hora_sweep'][sweep]), int(vol['info']['min_sweep'][sweep]), int(vol['info']['seg_sweep'][sweep]))   #Fecha y Hora Inicial del Barrido Sweep               
            fecha2 = datetime(int(vol['info']['ano_sweep_ini'][sweep]), int(vol['info']['mes_sweep_ini'][sweep]), int(vol['info']['dia_sweep_ini'][sweep]), int(vol['info']['hora_sweep_ini'][sweep]), int(vol['info']['min_sweep_ini'][sweep]), int(vol['info']['seg_sweep_ini'][sweep]))   #Fecha y Hora Inicial del Barrido Sweep                           
            diferencia = fecha1 - t_ref
            diferencia2 = fecha2 - t_ref
            radar.time['tiempo_inicial_sweep'][sweep]=diferencia.seconds
            radar.time['tiempo_inicial_sweep_0'][sweep]=diferencia2.seconds
        radar.time['tiempo_inicial_sweep_0'][0]=0 #De lo contrario queda un valor negativo

        if (debug):
            print radar.time['units']
            print radar.time['tiempo_inicial_sweep_0']
            print radar.time['tiempo_inicial_sweep']


        #--------------------------------------------------------------------------------
        # Tiempo de Inicio de cada RAYO respecto al tiempo de inicio del primer barrido
        #--------------------------------------------------------------------------------
        #   Se calcula el tiempo entre rayos aproximado.
        #   Como no hay dato de fin de barrido se toma el tiempo_inicial_sweep_0 del
        #   sgte barrido para calcular la duración del presente barrido. Puede que este
        #   bien calculado o que haya algunos segundos de diferencia, habría que chequear
        #   con INVAP como cargan los tiempos de inicio y final de cada barrido.
        #--------------------------------------------------------------------------------
        rayo=0
        for sweep in range (0, vol['info']['nsweeps']):  
            if sweep==(vol['info']['nsweeps']-1):       #Ultimo Sweep, no sabemos la duración del mismo. Copiamos el tiempo_entre_rayos anterior.
                tiempo_entre_rayos = (radar.time['tiempo_inicial_sweep'][sweep]-radar.time['tiempo_inicial_sweep'][sweep-1])/vol['info']['nrayos'][sweep]
                #print 'Sweep:',sweep,'Tiempo entre Rayos',tiempo_entre_rayos 

            else:
                tiempo_entre_rayos = (radar.time['tiempo_inicial_sweep_0'][sweep+1]-radar.time['tiempo_inicial_sweep'][sweep])/radar.rays_per_sweep['data'][sweep]
                #print 'Sweep_',sweep,'Tiempo entre Rayos',tiempo_entre_rayos 
            
            #Calculamos Tiempo Iniciales de Rayos
            for j in range (0, int(vol['info']['nrayos'][sweep])):
                radar.time['data'][rayo]=radar.time['tiempo_inicial_sweep'][sweep]+ j*tiempo_entre_rayos
                rayo=rayo+1    

#        time = np.zeros((radar.nrays))
#        for sweep in range (0, radar.nsweeps):          
#            if sweep==(radar.nsweeps-1):            #Ultimo Sweep, no sabemos la duración del mismo. Copiamos el tiempo_entre_rayos anterior.
#                tiempo_entre_rayos = (radar.time['tiempo_inicial_sweep'][sweep]-radar.time['tiempo_inicial_sweep'][sweep-1])/vol['info']['nrayos'][sweep]
#                for ray in range (0, radar.rays_per_sweep[sweep]):            
#                    time[ray+sweep*radar.rays_per_sweep[sweep]]=radar.time['tiempo_inicial_sweep'][sweep] + (ray * tiempo_entre_rayos) 
#
#            else:
#                tiempo_entre_rayos = (radar.time['tiempo_inicial_sweep_0'][sweep+1]-radar.time['tiempo_inicial_sweep'][sweep])/radar.rays_per_sweep['data'][sweep]
#                for ray in range (0, radar.rays_per_sweep[sweep]):            
#                   time[ray+sweep*radar.rays_per_sweep[sweep]]=radar.time['tiempo_inicial_sweep'][sweep] + (ray * tiempo_entre_rayos)                   
#        radar.time['data']=time 


        
        #-----------------------------------------------------------------
        #Parametros de Radar
        #-----------------------------------------------------------------
        #radar.instrument_parameters.keys()
        
        
        #-----------------------------------------------------------------
        #Calibracion de Radar
        #-----------------------------------------------------------------
        #radar2.radar_calibration.keys()
        #Si se tiene algun archivo con la calibracion/correcciones del radar acá se carga la info
        
        #-----------------------------------------------------------------
        # OTROS:
        # VER INFO: http://arm-doe.github.io/pyart-docs-travis/dev_reference/generated/pyart.core.radar.Radar.html#pyart.core.radar.Radar        
        #-----------------------------------------------------------------
        
        if (radar.metadata['instrument_name'][0:3] == 'RMA'):
#2          CHEQUEAR: ['data'] Probebalmente sea un vector
#            radar.ray_angle_res['data'] = 1   
#            radar.ray_angle_res['units'] = 'grados'     
#            radar.ray_angle_res['long_name'] = 'resolucion_angular_entre_rayos'
        
#2          #radar.rays_are_indexed      #VER Si es grilla con espaciado de 1 grado
                                     #    (dict or None) Indication of whether ray angles are indexed to a regular grid in each sweep. If not provided this attribute is set to None, indicating ray angle spacing is not determined.
        
#2          #radar.target_scan_rate      #VER Se podria cargar leyendo estrategia
                                     #    (dict or None) Intended scan rate for each sweep. If not provided this attribute is set to None, indicating this parameter is not available.
#2          #radar.sweep_mode            #VER (dict) Sweep mode for each mode in the volume scan.
#2          #radar.fixed_angle           #VER (dict) Target angle for thr sweep. Azimuth angle in RHI modes, elevation angle in all other modes.
#2          #radar.georefs_applied	       #VER (dict or None) Indicates whether the variables have had georeference calculation applied. Leading to Earth-centric azimuth and elevation angles.

            # SIN USO PARA RMAs
            radar.scan_rate = None          
            radar.antenna_transition = None 
            radar.drift     = None    #Para aviones
            radar.heading   = None    #Para aviones
            radar.pitch     = None    #Para aviones
            radar.roll      = None    #Para aviones
            radar.rotation  = None    #Para aviones
            radar.tilt      = None    #Para aviones

             
        #---------------------------------------------------------------------
        #CARGA de DATOS DE LOS PRODUCTOS
        #---------------------------------------------------------------------
        for nvolumen in range (0,len(volumenes)):                   
            
            vol=volumenes[nvolumen]
            #print "\n VOL_"+vol['info']['tipo_producto']+"\n nrayos: "+str(vol['info']['nrayos'])+"\n ngates: "+str(vol['info']['ngates'])+"\n gate_size: "+str(vol['info']['gate_size'])+"\n gate_offset: "+str(vol['info']['gate_offset'])                
            
            #Asignación de Unidades a Productos
            if (vol['info']['tipo_producto']=='TV'):
                vol['units']='dBZ'
            if (vol['info']['tipo_producto']=='TH'):
                vol['units']='dBZ'
            if (vol['info']['tipo_producto']=='ZDR'):
                vol['units']='dBZ'
            if (vol['info']['tipo_producto']=='CM'):
                vol['units']='dBZ'
            if (vol['info']['tipo_producto']=='PhiDP'):
                vol['units']='deg'
            if (vol['info']['tipo_producto']=='KDP'):
                vol['units']='deg/km'
            if (vol['info']['tipo_producto']=='VRAD'):
                vol['units']='m/s'
            if (vol['info']['tipo_producto']=='WRAD'):
               vol['units']='m/s'
            
            #...............................................................................
            # Corrección de Matrices de Productos con Dimensiones distintas a la de referencia.
            # Reordenamos los datos en función de la matriz de referencia que es la que tiene mayor rango
            #...............................................................................        
            if vol['info']['ngates'][0] != radar.ngates: 
    
                #print 'Matriz '+ vol['info']['tipo_producto'] +' corregida, dimensión distinta a la de referencia.'                    
                
                vol['data_org']=vol['data']
                vol['data']=np.ma.array(np.empty((radar.nrays,radar.ngates)))
                #vol['data']=np.empty((radar.nrays,radar.ngates))
                vol['data'][:] = np.NAN
                
                """Si estamos en esta instancia se debe a que el número de gates del nuevo volumen es menor que el de referencia.
                A continuación se comprueba si el offset de este nuevo producto es mayor que el offset de referencia. Si esto es cierto, entonces podemos calcular el gate inicial donde comenzar a copiar los datos. 
                En otras palabras se calcula el desplazamiento que debemos darle al nuevo producto a agregar en un arreglo nuevo de datos que tiene las mimas dimensiones que el de referencia) 
                La ecuación sería la siguiente: gate_inicial=(offset_volumen-offset_referencia) / gate_size
                También comprueba el el ancho de los gates sea el mismo sino no sería correcto desplazar el vector."""                    
                
                if vol['info']['gate_offset'][0] > radar.range['data'][0] and vol['info']['gate_size'][0] == (radar.range['data'][1]-radar.range['data'][0]):          
                    gate_inicial= (vol['info']['gate_offset'][0]-radar.range['data'][0]) / vol['info']['gate_size'][0]
                    gate_inicial=int(gate_inicial)
                    
                    for rayo in range (0,int(vol['info']['nrayos'][0])):
                        for gate in range (0,int(vol['info']['ngates'][0])):                         
                            vol['data'][rayo][gate_inicial+gate]=vol['data_org'][rayo][gate]
                
            #............................................................................... 
            #Carga de Datos de Productos Meteorológicos
            #...............................................................................
            tipo_producto=vol['info']['tipo_producto']
            del vol['info']   
            radar.add_field(tipo_producto, vol, replace_existing=True)
    
            ##Enmascara los Datos en función de los valores inválidos (NaN, INF, etc)
            radar.fields[tipo_producto]['data']=ma.masked_invalid(radar.fields[tipo_producto]['data'])
        
        #............................................................................... 
        # Guardar PyART Object en NetCDF file
        #...............................................................................
        if save_file==True:
            ##Copio los nombres de todos los productos cargados en el Objeto PyART
            #productos_agregados=''
            #for fichero in files:
            #    productos_agregados=productos_agregados+fichero.split('_')[3]+'_'
            
            #Elimino la extensión original del archivo leido y armo el nombre final por partes.
            fichero=nom_fichero.split('.')[0]
            fichero=fichero.split('_')[0]+'_'+fichero.split('_')[1]+'_'+fichero.split('_')[4]+'_'+fichero.split('_')[2]                       
            #fichero=fichero.split('_')[0]+'_'+fichero.split('_')[1]+'_'+fichero.split('_')[2]+'_'+productos_agregados+'_'+fichero.split('_')[4]           
            if path_out == 0: #Por defecto guarda los archivos de salida en los mismos directorios donde se encuentran los archivos de entrada            
                pyart.io.cfradial.write_cfradial(path+'/'+fichero+'.nc', radar, format='NETCDF4_CLASSIC', time_reference=False ,arm_time_variables=False)
            elif path_out !=0: #Guarda el archivo en la ruta especificada por el usuario
                pyart.io.cfradial.write_cfradial(path_out+'/'+fichero+'.nc', radar, format='NETCDF4_CLASSIC', time_reference=False ,arm_time_variables=False)
        
        return radar
    


    
def matlab_matrix_to_netcdf (path):
    matlab_matrix_to_PyARTobject(path, save_file=True)
        
        
        


def dec_bufr_file (bufr_filename=None):  
#    root_tables_bufr = './bufr/'
#    root_lib_dec_bufr = './bufr/libdecbufr.so'
    root_tables_bufr   = cf.root_tables_bufr
    root_lib_dec_bufr = cf.root_libdecbufr

    libdecbufr=cdll.LoadLibrary(root_lib_dec_bufr)         #Cargamos libdecbufr

    #==============================================================================
    # DECBUFR get_meta_data
    #==============================================================================
    get_meta_data = libdecbufr.get_meta_data
    get_meta_data.argtypes  = [c_char_p, c_char_p]
    get_meta_data.restype   = POINTER (meta_t)
    metadata = get_meta_data(bufr_filename, root_tables_bufr)

    vol_metadata = {}
    vol_metadata['year']  = metadata.contents.year
    vol_metadata['month'] = metadata.contents.month
    vol_metadata['day']   = metadata.contents.day
    vol_metadata['hour']  = metadata.contents.hour
    vol_metadata['min']   = metadata.contents.min
    vol_metadata['lat']   = metadata.contents.radar.lat
    vol_metadata['lon']   = metadata.contents.radar.lon
    vol_metadata['radar_height'] = metadata.contents.radar_height
    
    #==============================================================================
    # DECBUFR get_size_data
    #==============================================================================
    get_size_data= libdecbufr.get_size_data
    get_size_data.argtypes = [c_char_p, c_char_p]
    get_size_data.restype = c_int
    size_data=get_size_data(bufr_filename, root_tables_bufr)
    #print '\nSIZE DATA: ', size_data
    
    #==============================================================================
    # DECBUFR get_data_part
    #==============================================================================
    get_data_part= libdecbufr.get_data
    get_data_part.argtypes = [c_char_p, c_char_p]
    array=c_int*size_data
    get_data_part.restype = POINTER(array)
    vol_data=get_data_part(bufr_filename, root_tables_bufr)
    
    vol=list(vol_data.contents)         #Pasamos info a vector
    vol=np.asarray(vol)
    del vol_data
    #print '\nTAMAÑO VECTOR RECUPERADO: ',np.size(vol)
    
    
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
    
    sweeps_data=[]
    nsweeps = vol[0]
    vol_metadata['nsweeps'] = nsweeps
    
    u=1
    for sweep in range (nsweeps):
        barrido = {}
        barrido['compress_data'] = []

#        print 'First ','i:',u,'Valor:', vol[u]
        barrido['year_ini']         = vol[u];  u=u+1;
        barrido['month_ini']        = vol[u];  u=u+1;
        barrido['day_ini']          = vol[u];  u=u+1;
        barrido['hour_ini']         = vol[u];  u=u+1;
        barrido['min_ini']          = vol[u];  u=u+1;
        barrido['sec_ini']          = vol[u];  u=u+1;
        barrido['year']             = vol[u];  u=u+1;
        barrido['month']            = vol[u];  u=u+1;
        barrido['day']              = vol[u];  u=u+1;
        barrido['hour']             = vol[u];  u=u+1;
        barrido['min']              = vol[u];  u=u+1;
        barrido['sec']              = vol[u];  u=u+1;
        barrido['Type_of_product']  = vol[u];  u=u+1;
        barrido['elevation']        = vol[u];  u=u+1;
        barrido['ngates']           = vol[u];  u=u+1;
        barrido['range_size']       = vol[u];  u=u+1;
        barrido['range_offset']     = vol[u];  u=u+1;
        barrido['nrays']            = vol[u];  u=u+1;
        barrido['antenna_beam_az']  = vol[u];  u=u+2;
        barrido['Type of product']  = vol[u];  u=u+2;    
    
#        print 'MultiP','i:',u,'Valor:', vol[u]
        multi_pri = vol[u]; u=u+1   
        for i in range(multi_pri):
    
#            print 'MultiS','i:',u,'Valor:', vol[u]
            multi_sec = vol[u]; u=u+1;
            for j in range(multi_sec):
                if (vol[u]==99999):
                    barrido['compress_data'].append(255)
                else:
                    barrido['compress_data'].append(vol[u])   
                u=u+1
            
        barrido['compress_data'] = bytearray(barrido['compress_data'])
        sweeps_data.append(barrido)
    
    
    #==============================================================================
    # Descompresion 
    #==============================================================================
    import struct
    import zlib
    for sweep in range(nsweeps):
        sweeps_data[sweep]['data']=[]
        data_buf = buffer( sweeps_data[sweep]['compress_data'] )
        dec_data=zlib.decompress(data_buf)
        dec_data_bytes=bytearray(dec_data)
        
        j=0
        for i in range (np.size(dec_data_bytes)/8):
            sweeps_data[sweep]['data'].append(struct.unpack('d', dec_data_bytes[j:j+8])[0])
            j=j+8
           

    #==============================================================================
    # Adecuacion de los Datos
    #==============================================================================
    for sweep in range(nsweeps):
        #Enmascarado los valores 'Missing Values'
        sweeps_data[sweep]['data']= np.ma.masked_equal(sweeps_data[sweep]['data'], -1.797693134862315708e+308)  

        #Redimensionamos los datos de los barridos en un vector con dimensiones nrays*nagates
        sweeps_data[sweep]['data']= np.reshape(sweeps_data[sweep]['data'], (sweeps_data[sweep]['nrays'] , sweeps_data[sweep]['ngates']))    
    
    #Reacomodo la info para formato PyART
    vol_data=sweeps_data[0]['data']       #todos los haces de los n barridos forman un solo vector.
    for sweep in range(nsweeps-1):          
        vol_data=np.concatenate((vol_data,sweeps_data[sweep+1]['data']),axis=0)



    return vol_metadata, sweeps_data, vol_data




        
        
        

def get_vol_data_from_bufr_files (path=None, debug=False):      
    volumenes=[]
    sweep_inicial=0
    
    #==============================================================================
    # CARGAMOS INFO DE BUFR FILES
    #==============================================================================   
    lstDir = os.walk(path)   
    for root, dirs, files in lstDir:  
        for filename in files:
            if(filename.endswith('.BUFR')):

                vol={}  #Inicializo  
            
                #Carga metadata vol, sweeps y data de las variables polarimetricas
                [meta_data_vol, meta_data_sweeps, vol_data ] = dec_bufr_file(bufr_filename=path+filename)
                vol['metadata_vol']=meta_data_vol
                vol['metadata_vol']['radar_name'] = filename.split('_')[0]
                vol['metadata_vol']['estrategia_nombre'] = filename.split('_')[1]
                vol['metadata_vol']['estrategia_nvol'] = filename.split('_')[2]
                vol['metadata_vol']['tipo_producto'] = filename.split('_')[3]    #Nombre del field a descomprimir                
                vol['metadata_vol']['filename'] = filename
                vol['metadata_sweeps']=meta_data_sweeps    

                vol['data']=vol_data
                
                
                if debug:
                    print 'CARGANDO '+ vol['metadata_vol']['tipo_producto']
 
                #--------------------------------------------------------------------------------
                # Estrategia 9005
                #--------------------------------------------------------------------------------
                #   Parche momentaneo para Estrategia 9005. El 4to volumen tiene los 2 ultimos 
                #   barridos con dimensiones distintas a las primeras. Para procesar esto se debe 
                #   actualizar la version completa del conversor, no solo esta parte
                #---------------------------------------------------------------------------------    
                #......................................................................................
                # DATOS VOLUMEN: 
                #   Reordenamos los datos en función del formato esperado por PyART. 
                #   Formato esperado: nrayostotales_x_n_gates
                #......................................................................................  
                #! IMPLEMENTAR
                
                #    if (file_name.split('_')[1] == '9005'):
                #        if (debug):
                #            print 'Estrategia 9005'        
                #        
                #        num_volumen = file_name.split('_')[2]    
                #        if num_volumen != '04':  
                #            vol['data'] = vol['matriz_datos'][0][0]
                #            for i in range(1,(vol['matriz_datos'][0].shape[0])):
                #                vol['data']=np.concatenate((vol['data'], vol['matriz_datos'][0][i]), axis=0)    
                #        elif num_volumen == '04':
                #            vol['data']=np.concatenate((vol['matriz_datos'][0][5],vol['matriz_datos'][0][6]), axis=0)    #copiio solo los ultimos 2 barridos (no son de calibración, tienen info util)
                #            
                #            vol['info_barridos'] = [vol['info_barridos'][5],vol['info_barridos'][6]]
                #            vol['info_volumen'][0][0][9][0][0]=2 #nseeps
                #


                #==============================================================================
                # INFO VOLUMEN
                #==============================================================================
                #........................................................................
                #Info de Instrumento y Estrategia
                #........................................................................
                vol['info']={}
                vol['info']['estrategia']={}
                vol['info']['metadata']={}
                
                
                vol['info']['nombre_radar']                 = vol['metadata_vol']['radar_name']
                vol['info']['estrategia']['nombre']         = vol['metadata_vol']['estrategia_nombre']
                vol['info']['estrategia']['volume_number']  = vol['metadata_vol']['estrategia_nvol']
                vol['info']['tipo_producto']                = vol['metadata_vol']['tipo_producto']  
                vol['info']['filename']                     = vol['metadata_vol']['filename']  
                
                #........................................................................
                #Carga de Info General del Volumen
                #........................................................................
                vol['info']['ano_vol']  = vol['metadata_vol']['year']
                vol['info']['mes_vol']  = vol['metadata_vol']['month']
                vol['info']['dia_vol']  = vol['metadata_vol']['day']
                vol['info']['hora_vol'] = vol['metadata_vol']['hour']
                vol['info']['min_vol']  = vol['metadata_vol']['min']
                vol['info']['lat']      = vol['metadata_vol']['lat']
                vol['info']['lon']      = vol['metadata_vol']['lon']
                vol['info']['altura']   = vol['metadata_vol']['radar_height']
                vol['info']['nsweeps']  = vol['metadata_vol']['nsweeps']
                
                #........................................................................
                #Carga de Info de Barridos
                #........................................................................
                nsweeps=vol['info']['nsweeps']
                vol['info']['ano_sweep_ini']    = np.zeros(nsweeps); 
                vol['info']['mes_sweep_ini']    = np.zeros(nsweeps); 
                vol['info']['dia_sweep_ini']    = np.zeros(nsweeps); 
                vol['info']['hora_sweep_ini']   = np.zeros(nsweeps); 
                vol['info']['min_sweep_ini']    = np.zeros(nsweeps);
                vol['info']['seg_sweep_ini']    = np.zeros(nsweeps);
                vol['info']['ano_sweep']        = np.zeros(nsweeps);
                vol['info']['mes_sweep']        = np.zeros(nsweeps); 
                vol['info']['dia_sweep']        = np.zeros(nsweeps); 
                vol['info']['hora_sweep']       = np.zeros(nsweeps); 
                vol['info']['min_sweep']        = np.zeros(nsweeps); 
                vol['info']['seg_sweep']        = np.zeros(nsweeps)
                vol['info']['elevaciones']      = np.zeros(nsweeps)
                vol['info']['ngates']           = np.zeros(nsweeps); 
                vol['info']['gate_size']        = np.zeros(nsweeps); 
                vol['info']['gate_offset']      = np.zeros(nsweeps)
                vol['info']['nrayos']           = np.zeros(nsweeps); 
                vol['info']['rayo_inicial']     = np.zeros(nsweeps)
                
                for sweep in range (0,nsweeps):
                    vol['info']['ano_sweep_ini'] [sweep] = int(vol['metadata_sweeps'][sweep]['year_ini'])
                    vol['info']['mes_sweep_ini'] [sweep] = int(vol['metadata_sweeps'][sweep]['month_ini'])
                    vol['info']['dia_sweep_ini'] [sweep] = int(vol['metadata_sweeps'][sweep]['day_ini'])
                    vol['info']['hora_sweep_ini'][sweep] = int(vol['metadata_sweeps'][sweep]['hour_ini'])
                    vol['info']['min_sweep_ini'] [sweep] = int(vol['metadata_sweeps'][sweep]['min_ini'])
                    vol['info']['seg_sweep_ini'] [sweep] = int(vol['metadata_sweeps'][sweep]['sec_ini'])
                
                    vol['info']['ano_sweep']     [sweep] = int(vol['metadata_sweeps'][sweep]['year'])
                    vol['info']['mes_sweep']     [sweep] = int(vol['metadata_sweeps'][sweep]['month'])
                    vol['info']['dia_sweep']     [sweep] = int(vol['metadata_sweeps'][sweep]['day'])
                    vol['info']['hora_sweep']    [sweep] = int(vol['metadata_sweeps'][sweep]['hour'])
                    vol['info']['min_sweep']     [sweep] = int(vol['metadata_sweeps'][sweep]['min'])
                    vol['info']['seg_sweep']     [sweep] = int(vol['metadata_sweeps'][sweep]['sec'])
                
                    vol['info']['elevaciones']   [sweep] = int(vol['metadata_sweeps'][sweep]['elevation'])
                    vol['info']['ngates']        [sweep] = int(vol['metadata_sweeps'][sweep]['ngates'])
                    vol['info']['gate_size']     [sweep] = int(vol['metadata_sweeps'][sweep]['range_size'])
                    vol['info']['gate_offset']   [sweep] = int(vol['metadata_sweeps'][sweep]['range_offset'])
                    vol['info']['nrayos']        [sweep] = int(vol['metadata_sweeps'][sweep]['nrays'])
                    vol['info']['rayo_inicial']  [sweep] = int(vol['metadata_sweeps'][sweep]['antenna_beam_az'])
                    
                    
                    
                #........................................................................
                #Info para Metadata
                #........................................................................
                vol['info']['metadata']['comment']             ='-'
                vol['info']['metadata']['instrument_type']     ='Radar'
                vol['info']['metadata']['site_name']           ='-'
                vol['info']['metadata']['Sub_conventions']     ='-'
                vol['info']['metadata']['references']          ='-'
                vol['info']['metadata']['volume_number']       = vol['info']['estrategia']['volume_number']
                vol['info']['metadata']['scan_id']             = vol['info']['estrategia']['nombre']
                vol['info']['metadata']['title']               ='-'
                vol['info']['metadata']['source']              ='-'
                vol['info']['metadata']['version']             ='-'
                vol['info']['metadata']['instrument_name']     = vol['info']['nombre_radar']
                vol['info']['metadata']['ray_times_increase']  ='-'
                vol['info']['metadata']['platform_is_mobile']  ='false'
                vol['info']['metadata']['driver']              ='-'
                vol['info']['metadata']['institution']         ='SiNaRaMe'
                vol['info']['metadata']['n_gates_vary']        ='-'
                vol['info']['metadata']['primary_axis']        ='-'
                vol['info']['metadata']['created']             ='Fecha:'+str(int(vol['info']['dia_sweep'][0]))+'/'+str(int(vol['info']['mes_sweep'][0]))+'/'+str(int(vol['info']['ano_sweep'][0]))+' Hora:'+str(int(vol['info']['hora_sweep'][0]))+':'+str(int(vol['info']['min_sweep'][0]))+':'+str(int(vol['info']['seg_sweep'][0]))
                vol['info']['metadata']['scan_name']           ='-'
                vol['info']['metadata']['author']              ='Grupo Radar Cordoba (GRC) - Extractor/Conversor de Datos de Radar '
                vol['info']['metadata']['Conventions']         ='-'
                vol['info']['metadata']['platform_type']       ='Base Fija'
                vol['info']['metadata']['history']             ='-'    
                
                #........................................................................
                #Limpiamos el diccionario
                #........................................................................
                del vol['metadata_sweeps']    
                del vol['metadata_vol']    
    


               #---------------------------------------------------------------------            
                #CHEQUEA COMPATIBILIDAD DEL VOLUMEN_PRODUCTO (segun lo implementado actualmente)
                #---------------------------------------------------------------------
                for sweep in range (1, vol['info']['nsweeps']):            
                    #Los nrayos de todos los sweeps deben ser iguales
                    if vol['info']['nrayos'][0] != vol['info']['nrayos'][sweep]:
                        raise ValueError ("ERROR: Volumen no soportado, número de rayos distintos entre sweeps")                    
                
                    #Los ngates de todos los sweeps deben ser iguales    
                    if vol['info']['ngates'][0] != vol['info']['ngates'][sweep]:
                        raise ValueError ("ERROR: Volumen no soportado, número de gates distintos entre sweeps")                    
                
                    #Los gate_offset de todos los sweeps deben ser iguales    
                    if vol['info']['gate_offset'][0] != vol['info']['gate_offset'][sweep]:
                        raise ValueError ("ERROR: Volumen no soportado, valores de gate_offset distintos entre sweeps")                    
    
                    #Los gate_offset de todos los sweeps deben ser iguales    
                    if vol['info']['gate_size'][0] != vol['info']['gate_size'][sweep]:
                       raise ValueError ("ERROR: Volumen no soportado, valores de gate_size  distintos entre sweeps")                    
                    
            
                #--------------------------------------------------------------------- 
                #Agregamos Volumen Chequeado a Lista Volumenes
                #--------------------------------------------------------------------- 
                volumenes.append(vol)
        
                #---------------------------------------------------------------------            
                #CHEQUEA que todos los VOLUMENES (PRODUCTOS) CARGADOS PERTENEZCAN a la mimas fecha y hora.
                #---------------------------------------------------------------------
                if sweep_inicial==0:            
                    fecha_vol=filename.split('_')[4]
                    sweep_inicial=1
                
                else:
                    if fecha_vol != filename.split('_')[4] :
                        raise ValueError ('ERROR: Los productos en el directorio no pertenecen al mismo Volumen')


    return volumenes



def bufr_to_PyARTobject(path, save_file=False, path_out=None, debug=False):   
    #---------------------------------------------------------------------            
    #CARGA LOS DATOS DEL VOLUMEN
    #---------------------------------------------------------------------
    volumenes = {}
    volumenes = get_vol_data_from_bufr_files (path=path, debug=debug)
      

    if volumenes:                
        #---------------------------------------------------------------------            
        #DETECTA PRODUCTO DE REFERENCIA (aquel que tiene el range mas lejano)
        #---------------------------------------------------------------------
        rango_maximo=np.zeros(len(volumenes))
        for nvolumen in range (0,len(volumenes)):
            rango_maximo[nvolumen]=volumenes[nvolumen]['info']['gate_offset'][0]+(volumenes[nvolumen]['info']['gate_size'][0]*volumenes[nvolumen]['info']['ngates'][0])      
        nreferencia=np.argmax(rango_maximo) #retorna el índice del máximo valor en el array.
                
        #---------------------------------------------------------------------
        #CARGA DE DATOS e INFO del Volumen de Referencia
        #---------------------------------------------------------------------           
        vol=volumenes[nreferencia]  
        ref_filename = vol['info']['filename']
         
        #......................................................
        #Creación del Objeto PyART
        #LIMITACION: esta forma de crear los archivos impone la limitación 
        #que todos los barridos debe tener el mismo número de gates y rayos.            
        #......................................................
        radar = pyart.testing.make_empty_ppi_radar(vol['info']['ngates'][0], vol['info']['nrayos'][0], vol['info']['nsweeps'])
        
        #......................................................
        #Carga de Datos del Producto de Referencia en Objeto-PyART
        #......................................................
        radar.add_field(vol['info']['tipo_producto'], vol, replace_existing=True)
                      
        #......................................................
        #Carga de Campos de Informacion General en Objeto-PyART
        #......................................................
        #----------------------------------------------------------------- 
        #Altura del Radar
        #-----------------------------------------------------------------
        radar.altitude['data']=np.ndarray(1)
        radar.altitude['data'][0]=vol['info']['altura']
        radar.altitude['units']='metros'
        radar.altitude['long_name']='altitud'
        radar.altitude['possitive:']='arriba'
        radar.altitude['_fillValue']=-9999.0

#2      Chequear que esta sea la estructura para este campo, lo copie de altitude   
#2      Chequear si altura de RMAs, incluyen las torres. Sino cargar tabla con altura de torres y restar a altitud esa altura y ponersela a esta
#        radar.altitude_agl['data']=np.ndarray(1)
#        radar.altitude_agl['data'][0]=0
#        radar.altitude_agl['units']='metros'
#        radar.altitude_agl['long_name']='altitud_sobre_nivel_del_suelo'
#        radar.altitude_agl['possitive:']='arriba'
#        radar.altitude_agl['_fillValue']=-9999.0


        #-----------------------------------------------------------------
        #Azimuth / Elevación / Fixed_angle / 
        #-----------------------------------------------------------------
        z=0
        for i in range (0, vol['info']['nsweeps']):
            
            for j in range (0, int(vol['info']['nrayos'][i])):
                radar.azimuth['data'][z]=j
                radar.elevation['data'][z]=vol['info']['elevaciones'][i]
                z=z+1
            
            radar.fixed_angle['data'][i]=vol['info']['elevaciones'][i]
        
        #-----------------------------------------------------------------
        #Coordenadas Geográficas
        #-----------------------------------------------------------------
        radar.latitude['data']=np.ndarray(1)
        radar.latitude['data'][0]=vol['info']['lat']
        radar.latitude['units']='grados'
        radar.latitude['long_name']='latitud'
        radar.latitude['_fillValue']=-9999.0
        
        radar.longitude['data']=np.ndarray(1)
        radar.longitude['data'][0]=vol['info']['lon']
        radar.longitude['units']='grados'
        radar.longitude['long_name']='longitud'
        radar.longitude['_fillValue']=-9999.0
        
        #-----------------------------------------------------------------
        #Rango
        #-----------------------------------------------------------------
#!      #REVISAR otros archivos de radar si comienzan en 0 o en offset
        
        radar.range['data'][0]=vol['info']['gate_offset'][0] #inicia en el gate_offset
        for i in range (1, int(vol['info']['ngates'][0])):
            radar.range['data'][i]= radar.range['data'][i-1]+vol['info']['gate_size'][0]
        
        radar.range['meters_between_gates']=vol['info']['gate_size'][0]
        radar.range['meters_to_center_of_first_gate']=vol['info']['gate_offset'][0]
        
        
        
        #-----------------------------------------------------------------
        #MetaData
        #-----------------------------------------------------------------
        radar.metadata['comment']=vol['info']['metadata']['comment']                        #Nada
        radar.metadata['instrument_type']=vol['info']['metadata']['instrument_type']        #'Radar'
#2
        radar.metadata['site_name']=vol['info']['metadata']['site_name']                    #Nada       #PROX: cargar listado de sitios 
        radar.metadata['Sub_conventions']=vol['info']['metadata']['Sub_conventions']        #Nada
        radar.metadata['references']=vol['info']['metadata']['references']                  #Nada
        radar.metadata['volume_number']=vol['info']['metadata']['volume_number']            #Volumen dentro de Estrategia
        radar.metadata['scan_id']=vol['info']['metadata']['scan_id']                        #Estrategia
        radar.metadata['title']=vol['info']['metadata']['title']                            #Nada
        radar.metadata['source']=vol['info']['metadata']['source']                          #Nada
        radar.metadata['version']=vol['info']['metadata']['version']                        #Nada
        radar.metadata['instrument_name']=vol['info']['metadata']['instrument_name']        #Nombre Radar
#2
        radar.metadata['ray_times_increase']=vol['info']['metadata']['ray_times_increase']  #Nada       #PROX: Podria decir:'Variable de Barrido a Barrido' o leerla de estrategia y cargar vector con info
        radar.metadata['platform_is_mobile']=vol['info']['metadata']['platform_is_mobile']  #False  
        radar.metadata['driver']=vol['info']['metadata']['driver']                          #Nada
        radar.metadata['institution']=vol['info']['metadata']['institution']                #'SiNaRaMe'
        radar.metadata['n_gates_vary']=vol['info']['metadata']['n_gates_vary']              #'False'            
        radar.metadata['primary_axis']=vol['info']['metadata']['primary_axis']              #Nada
#!
        radar.metadata['created']=vol['info']['metadata']['created']                        #Fecha      #Chequear pq es incongruente la fecha
#2
        radar.metadata['scan_name']=vol['info']['metadata']['scan_name']                    #Nada       #PROX: leer de estrategia el tipo de procesamiento
        radar.metadata['author']=vol['info']['metadata']['author']                          #'Grupo Radar Cordoba (GRaC) - Extractor/Conversor de Datos de Radar' 
        radar.metadata['Conventions']=vol['info']['metadata']['Conventions']                #Nada
        radar.metadata['platform_type']=vol['info']['metadata']['platform_type']            #'Base Fija'
        radar.metadata['history']=vol['info']['metadata']['history']                        #Nada
        
        #*************************************************************************************************
        #TIEMPO
        #*************************************************************************************************
        radar.time['comment']='tiempos relativos al tiempo de inicio del primer barrido del volumen'
        radar.time['long_name']='tiempo en segundos desde inicio del primer barrido del volumen'
        radar.time['standard_name']='tiempo'
        radar.time['units']='seconds since '+str(int(vol['info']['ano_sweep'][0]))+'-'+str(int(vol['info']['mes_sweep'][0]))+'-'+str(int(vol['info']['dia_sweep'][0]))+'T'+str(int(vol['info']['hora_sweep'][0]))+':'+str(int(vol['info']['min_sweep'][0]))+':'+str(int(vol['info']['seg_sweep'][0]))+'Z'
        
        
        #--------------------------------------------------------------------------------
        #Tiempos Iniciales de Barridos (respecto al Tiempo de Inicio del Primer Barrido)
        #--------------------------------------------------------------------------------
        radar.time['tiempo_inicial_sweep']=np.zeros(vol['info']['nsweeps'])
        radar.time['tiempo_inicial_sweep_0']=np.zeros(vol['info']['nsweeps'])
        t_ref = datetime(int(vol['info']['ano_sweep'][0]), int(vol['info']['mes_sweep'][0]), int(vol['info']['dia_sweep'][0]), int(vol['info']['hora_sweep'][0]), int(vol['info']['min_sweep'][0]), int(vol['info']['seg_sweep'][0]))   #Fecha y Hora Inicial del Barrido 0. Lo hacemos referencia, todo el resto estarán referenciados por este.        
        
        for sweep in range (0, vol['info']['nsweeps']):
            fecha1 = datetime(int(vol['info']['ano_sweep'][sweep]), int(vol['info']['mes_sweep'][sweep]), int(vol['info']['dia_sweep'][sweep]), int(vol['info']['hora_sweep'][sweep]), int(vol['info']['min_sweep'][sweep]), int(vol['info']['seg_sweep'][sweep]))   #Fecha y Hora Inicial del Barrido Sweep               
            fecha2 = datetime(int(vol['info']['ano_sweep_ini'][sweep]), int(vol['info']['mes_sweep_ini'][sweep]), int(vol['info']['dia_sweep_ini'][sweep]), int(vol['info']['hora_sweep_ini'][sweep]), int(vol['info']['min_sweep_ini'][sweep]), int(vol['info']['seg_sweep_ini'][sweep]))   #Fecha y Hora Inicial del Barrido Sweep                           
            diferencia = fecha1 - t_ref
            diferencia2 = fecha2 - t_ref
            radar.time['tiempo_inicial_sweep'][sweep]=diferencia.seconds
            radar.time['tiempo_inicial_sweep_0'][sweep]=diferencia2.seconds
        radar.time['tiempo_inicial_sweep_0'][0]=0 #De lo contrario queda un valor negativo

        if (debug):
            print radar.time['units']
            print radar.time['tiempo_inicial_sweep_0']
            print radar.time['tiempo_inicial_sweep']


        #--------------------------------------------------------------------------------
        # Tiempo de Inicio de cada RAYO respecto al tiempo de inicio del primer barrido
        #--------------------------------------------------------------------------------
        #   Se calcula el tiempo entre rayos aproximado.
        #   Como no hay dato de fin de barrido se toma el tiempo_inicial_sweep_0 del
        #   sgte barrido para calcular la duración del presente barrido. Puede que este
        #   bien calculado o que haya algunos segundos de diferencia, habría que chequear
        #   con INVAP como cargan los tiempos de inicio y final de cada barrido.
        #--------------------------------------------------------------------------------
        rayo=0
        for sweep in range (0, vol['info']['nsweeps']):  
            if sweep==(vol['info']['nsweeps']-1):       #Ultimo Sweep, no sabemos la duración del mismo. Copiamos el tiempo_entre_rayos anterior.
                tiempo_entre_rayos = (radar.time['tiempo_inicial_sweep'][sweep]-radar.time['tiempo_inicial_sweep'][sweep-1])/vol['info']['nrayos'][sweep]
                #print 'Sweep:',sweep,'Tiempo entre Rayos',tiempo_entre_rayos 

            else:
                tiempo_entre_rayos = (radar.time['tiempo_inicial_sweep_0'][sweep+1]-radar.time['tiempo_inicial_sweep'][sweep])/radar.rays_per_sweep['data'][sweep]
                #print 'Sweep_',sweep,'Tiempo entre Rayos',tiempo_entre_rayos 
            
            #Calculamos Tiempo Iniciales de Rayos
            for j in range (0, int(vol['info']['nrayos'][sweep])):
                radar.time['data'][rayo]=radar.time['tiempo_inicial_sweep'][sweep]+ j*tiempo_entre_rayos
                rayo=rayo+1    

#        time = np.zeros((radar.nrays))
#        for sweep in range (0, radar.nsweeps):          
#            if sweep==(radar.nsweeps-1):            #Ultimo Sweep, no sabemos la duración del mismo. Copiamos el tiempo_entre_rayos anterior.
#                tiempo_entre_rayos = (radar.time['tiempo_inicial_sweep'][sweep]-radar.time['tiempo_inicial_sweep'][sweep-1])/vol['info']['nrayos'][sweep]
#                for ray in range (0, radar.rays_per_sweep[sweep]):            
#                    time[ray+sweep*radar.rays_per_sweep[sweep]]=radar.time['tiempo_inicial_sweep'][sweep] + (ray * tiempo_entre_rayos) 
#
#            else:
#                tiempo_entre_rayos = (radar.time['tiempo_inicial_sweep_0'][sweep+1]-radar.time['tiempo_inicial_sweep'][sweep])/radar.rays_per_sweep['data'][sweep]
#                for ray in range (0, radar.rays_per_sweep[sweep]):            
#                   time[ray+sweep*radar.rays_per_sweep[sweep]]=radar.time['tiempo_inicial_sweep'][sweep] + (ray * tiempo_entre_rayos)                   
#        radar.time['data']=time 


        
        #-----------------------------------------------------------------
        #Parametros de Radar
        #-----------------------------------------------------------------
        #radar.instrument_parameters.keys()
        
        
        #-----------------------------------------------------------------
        #Calibracion de Radar
        #-----------------------------------------------------------------
        #radar2.radar_calibration.keys()
        #Si se tiene algun archivo con la calibracion/correcciones del radar acá se carga la info
        
        #-----------------------------------------------------------------
        # OTROS:
        # VER INFO: http://arm-doe.github.io/pyart-docs-travis/dev_reference/generated/pyart.core.radar.Radar.html#pyart.core.radar.Radar        
        #-----------------------------------------------------------------
        
        if (radar.metadata['instrument_name'][0:3] == 'RMA'):
#2          CHEQUEAR: ['data'] Probebalmente sea un vector
#            radar.ray_angle_res['data'] = 1   
#            radar.ray_angle_res['units'] = 'grados'     
#            radar.ray_angle_res['long_name'] = 'resolucion_angular_entre_rayos'
        
#2          #radar.rays_are_indexed      #VER Si es grilla con espaciado de 1 grado
                                     #    (dict or None) Indication of whether ray angles are indexed to a regular grid in each sweep. If not provided this attribute is set to None, indicating ray angle spacing is not determined.
        
#2          #radar.target_scan_rate      #VER Se podria cargar leyendo estrategia
                                     #    (dict or None) Intended scan rate for each sweep. If not provided this attribute is set to None, indicating this parameter is not available.
#2          #radar.sweep_mode            #VER (dict) Sweep mode for each mode in the volume scan.
#2          #radar.fixed_angle           #VER (dict) Target angle for thr sweep. Azimuth angle in RHI modes, elevation angle in all other modes.
#2          #radar.georefs_applied	       #VER (dict or None) Indicates whether the variables have had georeference calculation applied. Leading to Earth-centric azimuth and elevation angles.

            # SIN USO PARA RMAs
            radar.scan_rate = None          
            radar.antenna_transition = None 
            radar.drift     = None    #Para aviones
            radar.heading   = None    #Para aviones
            radar.pitch     = None    #Para aviones
            radar.roll      = None    #Para aviones
            radar.rotation  = None    #Para aviones
            radar.tilt      = None    #Para aviones

             
        #---------------------------------------------------------------------
        #CARGA de DATOS DE LOS PRODUCTOS
        #---------------------------------------------------------------------
        for nvolumen in range (0,len(volumenes)):                   
            
            vol=volumenes[nvolumen]
            #print "\n VOL_"+vol['info']['tipo_producto']+"\n nrayos: "+str(vol['info']['nrayos'])+"\n ngates: "+str(vol['info']['ngates'])+"\n gate_size: "+str(vol['info']['gate_size'])+"\n gate_offset: "+str(vol['info']['gate_offset'])                
            
            #Asignación de Unidades a Productos
            if (vol['info']['tipo_producto']=='TV'):
                vol['units']='dBZ'
            if (vol['info']['tipo_producto']=='TH'):
                vol['units']='dBZ'
            if (vol['info']['tipo_producto']=='ZDR'):
                vol['units']='dBZ'
            if (vol['info']['tipo_producto']=='CM'):
                vol['units']='dBZ'
            if (vol['info']['tipo_producto']=='PhiDP'):
                vol['units']='deg'
            if (vol['info']['tipo_producto']=='KDP'):
                vol['units']='deg/km'
            if (vol['info']['tipo_producto']=='VRAD'):
                vol['units']='m/s'
            if (vol['info']['tipo_producto']=='WRAD'):
               vol['units']='m/s'
            
            #...............................................................................
            # Corrección de Matrices de Productos con Dimensiones distintas a la de referencia.
            # Reordenamos los datos en función de la matriz de referencia que es la que tiene mayor rango
            #...............................................................................        
            if vol['info']['ngates'][0] != radar.ngates: 
    
                if debug:
                    print vol['info']['tipo_producto'] +' corregido: , dimensión distinta a la de referencia.' 
                    print 'ngates Referencia',  radar.ngates
                    print 'ngates Producto', vol['info']['ngates'][0]
                    print 'Offset Referencia', vol['info']['gate_offset'][0]
                    print 'Offset Producto', radar.range['data'][0]
                    print 'Gate Size Referencia', radar.range['data'][1]-radar.range['data'][0]
                    print 'Gate Size Producto', vol['info']['gate_size'][0]

                
                vol['data_org']=vol['data'].copy()
                vol['data']=np.empty((radar.nrays,radar.ngates))
                vol['data'][:] = np.NAN                              
                #Si tienen el mismo offset inicial, copiamos los datos desde el primer elemento
                if vol['info']['gate_offset'][0] == radar.range['data'][0] and vol['info']['gate_size'][0] == (radar.range['data'][1]-radar.range['data'][0]):                             

                    for rayo in range (0,int(vol['info']['nrayos'][0])):
                        for gate in range (0,int(vol['info']['ngates'][0])):                         
                            vol['data'][rayo][gate]=vol['data_org'][rayo][gate]                                
                
                
                #Si estamos en esta instancia se debe a que el número de gates del nuevo volumen es menor que el de referencia.
                #A continuación se comprueba si el offset de este nuevo producto es mayor que el offset de referencia. Si esto es cierto, entonces podemos calcular el gate inicial donde comenzar a copiar los datos. 
                #En otras palabras se calcula el desplazamiento que debemos darle al nuevo producto a agregar en un arreglo nuevo de datos que tiene las mimas dimensiones que el de referencia) 
                #La ecuación sería la siguiente: gate_inicial=(offset_volumen-offset_referencia) / gate_size
                #También comprueba el el ancho de los gates sea el mismo sino no sería correcto desplazar el vector.
                
                #Si tienen un offset inicial distinto, copiamos los datos desplazados en el vector final
                elif vol['info']['gate_offset'][0] > radar.range['data'][0] and vol['info']['gate_size'][0] == (radar.range['data'][1]-radar.range['data'][0]):          
                    gate_inicial= (vol['info']['gate_offset'][0]-radar.range['data'][0]) / vol['info']['gate_size'][0]
                    gate_inicial=int(gate_inicial)
                    
                    for rayo in range (0,int(vol['info']['nrayos'][0])):
                        for gate in range (0,int(vol['info']['ngates'][0])):                         
                            vol['data'][rayo][gate_inicial+gate]=vol['data_org'][rayo][gate]
                
                else:
                    raise ValueError ('Error al intentar acomodar los productos meteorologicos.')
                
                
            #............................................................................... 
            #Carga de Datos de Productos Meteorológicos
            #...............................................................................
            tipo_producto=vol['info']['tipo_producto']
            del vol['info']   
            radar.add_field(tipo_producto, vol, replace_existing=True)
    
            ##Enmascara los Datos en función de los valores inválidos (NaN, INF, etc)
            radar.fields[tipo_producto]['data']=ma.masked_invalid(radar.fields[tipo_producto]['data'])
            radar.fields[tipo_producto]['data']=ma.masked_outside(radar.fields[tipo_producto]['data'], -100000, 100000)
            
        
        #............................................................................... 
        # Guardar PyART Object en NetCDF file
        #...............................................................................
        if save_file==True:
            ##Copio los nombres de todos los productos cargados en el Objeto PyART
            #productos_agregados=''
            #for fichero in files:
            #    productos_agregados=productos_agregados+fichero.split('_')[3]+'_'
            
            #Elimino la extensión original del archivo leido y armo el nombre final por partes.
            fichero=ref_filename.split('.')[0]
            fichero=fichero.split('_')[0]+'_'+fichero.split('_')[1]+'_'+fichero.split('_')[4]+'_'+fichero.split('_')[2]                       
            #fichero=fichero.split('_')[0]+'_'+fichero.split('_')[1]+'_'+fichero.split('_')[2]+'_'+productos_agregados+'_'+fichero.split('_')[4]           
            if path_out == 0: #Por defecto guarda los archivos de salida en los mismos directorios donde se encuentran los archivos de entrada            
                pyart.io.cfradial.write_cfradial(path+'/'+fichero+'.nc', radar, format='NETCDF4_CLASSIC', time_reference=False ,arm_time_variables=False)
            elif path_out !=0: #Guarda el archivo en la ruta especificada por el usuario
                pyart.io.cfradial.write_cfradial(path_out+'/'+fichero+'.nc', radar, format='NETCDF4_CLASSIC', time_reference=False ,arm_time_variables=False)
        
        return radar
    

  
def bufr_to_netcdf (path_in=None, path_out=None, debug=False):
    for dirName, subdirList, fileList in os.walk(path_in):  #recorre recursivamente todo el path, en cada iteración actualiza las variables
        if debug: 
            print('Directorio Actual: %s' % dirName)
        
        if fileList:    #si hay archivos en la carpeta intetamos convertirlos     
            bufr_to_PyARTobject(dirName+'/', save_file=True, path_out=path_out, debug=debug)    
            
    if debug:
        print ''
        print 'CONVERSION FINALIZADA'
        print ''


    
