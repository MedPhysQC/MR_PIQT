#!/usr/bin/env python
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# This code is an analysis module for WAD-QC 2.0: a server for automated 
# analysis of medical images for quality control.
#
# The WAD-QC Software can be found on 
# https://bitbucket.org/MedPhysNL/wadqc/wiki/Home
# 
#
# Changelog:
#   20200508: do not exit, raise error
#   20200508: dropping support for python2; dropping support for WAD-QC 1; toimage no longer exists in scipy.misc
#   20190426: Fix for matplotlib>3
#   20180418: treat DICOM Central_freq as float result
#   20170929: missing NEMA linearity result
#   20161220: remove class variables; remove testing stuff
#   20160802: sync with pywad1.0
#   20160622: removed adding limits (now part of analyzer)
#   20160620: remove quantity and units
#
# /QCMR_wadwrapper.py -c Config/mr_philips_achieva30_sHB_umcu_series.json -d TestSet/StudyAchieva30 -r results_achieva30.json

__version__ = '20200508'
__author__ = 'aschilham'

import os
# this will fail unless wad_qc is already installed
from wad_qc.module import pyWADinput
from wad_qc.modulelibs import wadwrapper_lib

if not 'MPLCONFIGDIR' in os.environ:
    import pkg_resources
    try:
        #only for matplotlib < 3 should we use the tmp work around, but it should be applied before importing matplotlib
        matplotlib_version = [int(v) for v in pkg_resources.get_distribution("matplotlib").version.split('.')]
        if matplotlib_version[0]<3:
            os.environ['MPLCONFIGDIR'] = "/tmp/.matplotlib" # if this folder already exists it must be accessible by the owner of WAD_Processor 
    except:
        os.environ['MPLCONFIGDIR'] = "/tmp/.matplotlib" # if this folder already exists it must be accessible by the owner of WAD_Processor 

import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.

try:
    import pydicom as dicom
except ImportError:
    import dicom
import QCMR_lib
import QCMR_constants as lit

def logTag():
    return "[QCMR_wadwrapper] "

# MODULE EXPECTS PYQTGRAPH DATA: X AND Y ARE TRANSPOSED!

##### Real functions
def qc_series(data, results, action):
    """
    QCMR_UMCU Checks: Philips PiQT reimplementen in python
      Uniformity (several ways)
      Geometrical deformation
      ArtifactLevel
      Signal-to-noise (several positions)
      Resolution (MTF, FTW, SliceWidth)

    Workflow:
        1. Read image or sequence
        2. Run test
        3. Build xml output
    """
    try:
        params = action['params']
    except KeyError:
        params = {}

    dcmInfile,pixeldataIn,dicomMode = wadwrapper_lib.prepareInput(data.series_filelist[0],headers_only=False,logTag=logTag())
    qclib = QCMR_lib.PiQT_QC()
    cs = QCMR_lib.PiQT_Struct(dcmInfile=dcmInfile,pixeldataIn=pixeldataIn,dicomMode=dicomMode,piqttest=None)
    cs.verbose = False

    ## id scanner
    idname = ""
    error = qclib.DetermineScanID(cs)
    if error == True or cs.scanID == lit.stUnknown:
        raise ValueError("{} ERROR! Cannot determine MR Scan ID".format(logTag()))

    ## 2. Run tests
    setname = cs.scanID

    piqttests = []
    if setname == 'QA1':
        piqttests = [ # seq test, "M", philips_slicenumber, echonumber, echotime,
            ("QA1_Uniformity",   "M",3,1, 30),
            ("QA1_Uniformity",   "M",3,2,100),
            ("QA1_Linearity" ,   "M",2,1, 30),
            ("QA1_SliceProfile", "M",4,1, 30),
            ("QA1_SliceProfile", "M",4,2,100),
            ("QA1_MTF",          "M",5,1, 30),
            ("QA1_MTF",          "M",5,2,100),
        ]
    elif setname== 'QA2':
        piqttests = [ # seq test, "M", philips_slicenumber, echonumber, echotime,
            ("QA2_Uniformity",   "M",2,1, 15),
            ("QA2_SliceProfile", "M",3,1, 15),
       ]
    elif setname == 'QA3':
        piqttests = [ # seq test, "M", philips_slicenumber, echonumber, echotime,
            ("QA3_Uniformity",   "M",1,1, 50),
            ("QA3_Uniformity",   "M",1,2,100),
            ("QA3_Uniformity",   "M",1,3,150),
        ]

    reportkeyvals = []
    for piqt in piqttests:
        print("[mrqc]",2,piqt)
        if "Uniformity" in piqt[0]:
            test = lit.stTestUniformity
            doTest = "SNR_ArtifactLevel_FloodFieldUniformity"
        if "Linearity" in piqt[0]:
            test = lit.stTestSpatialLinearity
            doTest = "SpatialLinearity"
        if "SliceProfile" in piqt[0]:
            test = lit.stTestSliceProfile
            doTest = "SliceProfile"
        if "MTF" in piqt[0]:
            test = lit.stTestMTF
            doTest = "MTF"

        cs = QCMR_lib.PiQT_Struct(dcmInfile,pixeldataIn,dicomMode,piqt)
        cs.verbose = None

        if "FloodFieldUniformity" in doTest: # FFU also contains SNR and ArtifactLevel
            error = qclib.FloodFieldUniformity(cs)
            if not error:
                import numpy as np
                idname = "_"+setname+make_idname(qclib,cs,cs.snr_slice)
                reportkeyvals.append( ("S/N (B)"+idname,cs.snr_SNB) )
                reportkeyvals.append( ("Art_Level"+idname,cs.artefact_ArtLevel) )

                reportkeyvals.append( ("T/C-20"+idname,cs.ffu_TCm20) )
                reportkeyvals.append( ("C-20/C-10"+idname,cs.ffu_Cm20Cm10) )
                reportkeyvals.append( ("C-10/C+10"+idname,cs.ffu_Cm10Cp10) )
                reportkeyvals.append( ("C+10/C+20"+idname,cs.ffu_Cp10Cp20) )
                reportkeyvals.append( ("C+20/Max"+idname,cs.ffu_Cp20MAX) )
                reportkeyvals.append( ("Rad 10%"+idname,cs.ffu_rad10) )
                reportkeyvals.append( ("Int_Unif"+idname,cs.ffu_lin_unif) )
                ## Build thumbnail
                filename = 'FFU'+idname+'.jpg' # Use jpg if a thumbnail is desired
                qclib.saveResultImage(cs,'FFU',filename)
                varname = 'FFU'+'_'+idname
                results.addObject(varname, filename)
                
        elif "ArtifactLevel" in doTest: # Artifact also contains SNR
            error = qclib.ArtifactLevel(cs)
            if not error:
                idname = "_"+setname+make_idname(qclib,cs,cs.snr_slice)
                reportkeyvals.append( ("S/N (B)"+idname,cs.snr_SNB) )
                reportkeyvals.append( ("Art_Level"+idname,cs.artefact_ArtLevel) )
        elif "SNR" in doTest:
            error = qclib.SNR(cs)
            if not error:
                idname = "_"+setname+make_idname(qclib,cs,cs.snr_slice)
                reportkeyvals.append( ("S/N (B)"+idname,cs.snr_SNB) )

        if "SpatialLinearity" in doTest:
            error = qclib.SpatialLinearity(cs)
            if not error:
                idname = "_"+setname+make_idname(qclib,cs,cs.lin_slice)
                reportkeyvals.append( ("phant_rot"+idname,cs.lin_phantomrotdeg) )
                reportkeyvals.append( ("m/p_angle"+idname,str(-360)) )
                reportkeyvals.append( ("size_hor"+idname,cs.lin_sizehor) )
                reportkeyvals.append( ("size_ver"+idname,cs.lin_sizever) )
                reportkeyvals.append( ("hor_int_av"+idname,cs.lin_intshiftavg[0]) )
                reportkeyvals.append( ("hor_int_dev"+idname,cs.lin_intshiftsdev[0]) )
                reportkeyvals.append( ("hor_max_right"+idname,cs.lin_shiftmax[0]) )
                reportkeyvals.append( ("hor_max_left"+idname,cs.lin_shiftmin[0]) )
                reportkeyvals.append( ("hor_diff_av"+idname,cs.lin_intdiffavg[0]) )
                reportkeyvals.append( ("hor_diff_dev"+idname,cs.lin_intdiffsdev[0]) )
                reportkeyvals.append( ("hor_max"+idname,cs.lin_intdiffmax[0]) )
                reportkeyvals.append( ("hor_min"+idname,cs.lin_intdiffmin[0]) )
                reportkeyvals.append( ("ver_int_av"+idname,cs.lin_intshiftavg[1]) )
                reportkeyvals.append( ("ver_int_dev"+idname,cs.lin_intshiftsdev[1]) )
                reportkeyvals.append( ("ver_max_up"+idname,cs.lin_shiftmax[1]) )
                reportkeyvals.append( ("ver_max_down"+idname,cs.lin_shiftmin[1]) )
                reportkeyvals.append( ("ver_diff_av"+idname,cs.lin_intdiffavg[1]) )
                reportkeyvals.append( ("ver_diff_dev"+idname,cs.lin_intdiffsdev[1]) )
                reportkeyvals.append( ("ver_max"+idname,cs.lin_intdiffmax[1]) )
                reportkeyvals.append( ("ver_min"+idname,cs.lin_intdiffmin[1]) )
                reportkeyvals.append( ("lin_NEMA"+idname,cs.lin_nema_max) )
                ## Build thumbnail
                filename = 'LIN'+idname+'.jpg' # Use jpg if a thumbnail is desired
                qclib.saveResultImage(cs,'LIN',filename)
                varname = 'LIN'+'_'+idname
                results.addObject(varname, filename)

        if "SliceProfile" in doTest:
            error = qclib.SliceProfile(cs)
            if not error:
                idname = "_"+setname+make_idname(qclib,cs,cs.sp_slice)
                reportkeyvals.append( ("Pos_shift"+idname,cs.sp_phantomshift) )
                reportkeyvals.append( ("FWHM"+idname,cs.sp_slicewidth_fwhm) )
                reportkeyvals.append( ("FWTM"+idname,cs.sp_slicewidth_fwtm) )
                reportkeyvals.append( ("Slice_int"+idname,cs.sp_slicewidth_lineint) )
                reportkeyvals.append( ("Angle"+idname,cs.sp_phantomzangledeg) )
                reportkeyvals.append( ("phant_rot"+idname,cs.sp_phantomrotdeg) )
                reportkeyvals.append( ("Phase_Shift"+idname,cs.sp_phaseshift) )
                ## Build thumbnail
                filename = 'SLP'+idname+'.jpg' # Use jpg if a thumbnail is desired
                qclib.saveResultImage(cs,'SLP',filename)
                varname = 'SLP'+'_'+idname
                results.addObject(varname, filename)

        if "MTF" in doTest:
            error = qclib.MTF(cs)
            if not error:
                idname = "_"+setname+make_idname(qclib,cs,cs.mtf_slice)
                reportkeyvals.append( ("Hor_pxl_size"+idname,cs.mtf_pixelsize[0]) )
                reportkeyvals.append( ("Ver_pxl_size"+idname,cs.mtf_pixelsize[1]) )
                ## Build thumbnail
                filename = 'MTF'+idname+'.jpg' # Use jpg if a thumbnail is desired
                qclib.saveResultImage(cs,'MTF',filename)
                varname = 'MTF'+'_'+idname
                results.addObject(varname, filename)

        if error:
            raise ValueError("{} ERROR! processing error in {} {}".format(logTag(),piqt,doTest))

    for key,val in reportkeyvals:
        results.addFloat(key, val)

def acqdatetime_series(data, results, action):
    """
    Read acqdatetime from dicomheaders and write to IQC database

    Workflow:
        1. Read only headers
    """
    try:
        params = action['params']
    except KeyError:
        params = {}

    ## 1. read only headers
    dcmInfile = dicom.read_file(data.series_filelist[0][0], stop_before_pixels=True)

    dt = wadwrapper_lib.acqdatetime_series(dcmInfile)

    results.addDateTime('AcquisitionDateTime', dt) 

def header_series(data, results, action):
    """
    Read selected dicomfields and write to IQC database

    Workflow:
        1. Run tests
        2. Build xml output
    """
    try:
        params = action['params']
    except KeyError:
        params = {}

    info = 'dicom'
    dcmInfile,pixeldataIn,dicomMode = wadwrapper_lib.prepareInput(data.series_filelist[0],headers_only=True,logTag=logTag())
    qclib = QCMR_lib.PiQT_QC()
    cs = QCMR_lib.PiQT_Struct(dcmInfile=dcmInfile,pixeldataIn=pixeldataIn,dicomMode=dicomMode,piqttest=None)
    cs.verbose = False
    
    ## id scanner
    idname = ""
    error = qclib.DetermineScanID(cs)
    if error == True or cs.scanID == lit.stUnknown:
        raise ValueError("{} ERROR! Cannot determine MR Scan ID".format(logTag()))

    ## 1. Run tests
    setname = cs.scanID

    piqttests = []
    if setname == 'QA1':
        piqttests = [ # seq test, "M", philips_slicenumber, echonumber, echotime,
            ("QA1_Uniformity",   "M",3,1, 30),
            ("QA1_Uniformity",   "M",3,2,100),
            ("QA1_Linearity" ,   "M",2,1, 30),
            ("QA1_SliceProfile", "M",4,1, 30),
            ("QA1_SliceProfile", "M",4,2,100),
            ("QA1_MTF",          "M",5,1, 30),
            ("QA1_MTF",          "M",5,2,100),
        ]
    elif setname== 'QA2':
        piqttests = [ # seq test, "M", philips_slicenumber, echonumber, echotime,
            ("QA2_Uniformity",   "M",2,1, 15),
            ("QA2_SliceProfile", "M",3,1, 15),
       ]
    elif setname == 'QA3':
        piqttests = [ # seq test, "M", philips_slicenumber, echonumber, echotime,
            ("QA3_Uniformity",   "M",1,1, 50),
            ("QA3_Uniformity",   "M",1,2,100),
            ("QA3_Uniformity",   "M",1,3,150),
        ]

    reportkeyvals = []
    for piqt in piqttests:
        cs = QCMR_lib.PiQT_Struct(dcmInfile,pixeldataIn,dicomMode,piqt)
        cs.verbose = None

        ## 1b. Run tests
        sliceno = qclib.ImageSliceNumber(cs,piqt)
        if sliceno <0:
            msg = "{} [mrheader]: {} not available for given image".format(logTag(), piqt)
            print(msg)
            raise ValueError(msg)

        dicominfo = qclib.DICOMInfo(cs,info,sliceno)
        if len(dicominfo) >0:
            idname = "_"+setname+make_idname(qclib,cs,sliceno)
            for di in dicominfo:
                reportkeyvals.append( (di[0]+idname,str(di[1])) )

    ## 2. Build xml output
    floatlist = [
        'Central_freq'
    ]
    # plugionversion is newly added in for this plugin since pywad2
    varname = 'pluginversion'+idname
    results.addString(varname, str(qclib.qcversion)) # do use default from config
    for key,val in reportkeyvals:
        is_float = False
        for fl in floatlist:
            if key.startswith(fl):
                is_float = True
                break

        if is_float:
            try:
                val2 = float(val)
            except ValueError:
                val2 = -1
            results.addFloat(key, val2)
        else:
            val2 = "".join([x if ord(x) < 128 else '?' for x in val]) #ignore non-ascii 
            results.addString(key, str(val2)[:min(len(str(val)),100)]) # do use default from config
    
def make_idname(qclib,cs,sliceno):
    idname = ''
    idfields = [
        ["2001,100a", "Slice_No"], # SliceLocation/slicespacing
        ["0018,0086", "Echo_No"], # 1
        ["0018,0081", "Echo_Time"], # 50
    ]
    for df in idfields:
        idname += '_'+str(int(qclib.readDICOMtag(cs,df[0],sliceno)))# int should not be needed but for MR7
    return idname


if __name__ == "__main__":
    data, results, config = pyWADinput()

    # read runtime parameters for module
    for name,action in config['actions'].items():
        if name == 'acqdatetime':
            acqdatetime_series(data, results, action)

        elif name == 'header_series':
            header_series(data, results, action)
        
        elif name == 'qc_series':
            qc_series(data, results, action)

    #results.limits["minlowhighmax"]["mydynamicresult"] = [1,2,3,4]

    results.write()
