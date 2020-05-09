# -*- coding: utf-8 -*-
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

"""
Warning: THIS MODULE EXPECTS PYQTGRAPH DATA: X AND Y ARE TRANSPOSED! And make sure rescaling is corrected!

TODO: Linearity : m/p angle
TODO: SliceProfile: phase shift
TODO: pixelsizes
Changelog:
    20200508: fix for "slice not found" because pydicom 2 adds quotes to pydicom.valuerep.IS
    20200508: dropping support for python2; dropping support for WAD-QC 1; toimage no longer exists in scipy.misc
    20181105: Added NEMA 2 image determination of SNR
    20180327: Removed some outdated comments.
    20171116: fix scipy version 1.0
    20170929: Missing transmit coil type tag
    20170109: Protocolnames are sometimes prefixed by WIP
    20161220: remove class variables; remove testing stuff
    20160902: sync with wad2.0; Unified pywad1.0 and wad2.0
    20151111: Added resultimages for all tests
    20151104: Many changes to fix QA1, QA2, QA3 agreement with Philips results; removed AffineTransform, used lowpass instead of movingaverage; changed signs
    20150629: Added feedback if MTF failed because of wrongly position phantom
    20150413: Gaussian fit with offset; fixed negative slice profile for Mxy
    20150204: Fixed missing SliceThickness for EnhancedDicom
    20141009: Moved strings for dicomModes to wadwrapper_lib
    20140930: Now also for enhanced DICOM; all DICOM read through wadwrapper_lib
    20140905: Force usage of Ramp for FWHM measurement (seems to be the case for MR8), regardless of slicethickness
    20140904: Fix for ImageSliceNumber; sending to dcm4chee changes some tags, and can add unexpected spaces.
    20140528: Initialize all items of PiQTStruct in __init__ (bug for gui)
    20140523: Fixed MTF and PhaseShift; removed global dcmFile, pixelData
    20140205: MTF implemented. Not correct result for pixelsizes (zeroes between left/right peak, or on right side of peak); possible mismatch angle (fixed 11.3 does better!)
    20140204: Correct slice number calculation; SP phantom shift and rotation and ramp angle; rescale images (done in GUI! but remember!); some phaseshift guestimate
    20131014: Corrected Slice Profile calculations (now using Schneiders1980)
    20131012: Finished Slice Profile
    20131011: Finished Linearity
    20131010: FFU calc of rad10 and rad20 by Euclidean distance transform
    20131009: Finished SNR; finished ArtLevel; finish FloodField Uniformity
"""
__version__ = '20200508'
__author__ = 'aschilham'

import numpy as np
import scipy.ndimage as scind
import matplotlib.pyplot as plt
import copy

try:
    # wad2.0 runs each module stand alone
    import QCMR_constants as lit
    import QCMR_math as mymath
except ImportError:
    from . import QCMR_constants as lit
    from . import QCMR_math as mymath

LOCALIMPORT = False
try: 
    # try local folder
    import wadwrapper_lib
    LOCALIMPORT = True
except ImportError:
    # try wad2.0 from system package wad_qc
    from wad_qc.modulelibs import wadwrapper_lib

# for image results
from PIL import Image # image from pillow is needed
from PIL import ImageDraw # imagedraw from pillow is needed, not pil
try:
    from scipy.misc import toimage
except (ImportError, AttributeError) as e:
    try:
        if LOCALIMPORT:
            from wadwrapper_lib import toimage as toimage
        else:
            from wad_qc.modulelibs.wadwrapper_lib import toimage as toimage
    except (ImportError, AttributeError) as e:
        msg = "Function 'toimage' cannot be found. Either downgrade scipy or upgrade WAD-QC."
        raise AttributeError("{}: {}".format(msg, e))

# sanity check: we need at least scipy 0.10.1 to avoid problems mixing PIL and Pillow
import scipy
scipy_version = [int(v) for v in scipy.__version__ .split('.')]
if scipy_version[0] == 0:
    if scipy_version[1]<10 or (scipy_version[1] == 10 and scipy_version[1]<1):
        raise RuntimeError("scipy version too old. Upgrade scipy to at least 0.10.1")

class PiQT_Struct:
    def __init__ (self,dcmInfile,pixeldataIn,dicomMode,piqttest):
        # input image
        self.dcmInfile = dcmInfile
        self.pixeldataIn = pixeldataIn
        self.dicomMode = dicomMode
        self.scanID = lit.stUnknown
        self.piqttest = piqttest # tuple (seq_test,imagetype,philipsslicenumber,echonumber,echotime,)
        self.verbose = False

        # for matlib plotting
        self.hasmadeplots = False

        # SNR 
        self.snr_means  = [] # average in Center, Background
        self.snr_stdevs = [] # Stdev in Center, Background
        self.snr_rois   = [] # xy roi definitions # format: x0,wid, yo,hei
        self.snr_slice  = -1
        self.snr_slice2 = -1
        self.snr_bkgrnd = -1
        self.snr_SNC = -1
        self.snr_SNB = -1
        self.snr_BsdB = -1

        # Artefact Level 
        self.artefact_max = None # Maximum mean value of ROI of 3*3 pixels in background of image. The edges of the image are masked.
        self.artefact_roi = [] # x0,y0,rad
        self.artefact_ArtLevel = -1

        # FloodField Uniformity
        self.ffu_Ntot     = None
        self.ffu_TCm20    = -1.
        self.ffu_Cm20Cm10 = -1.
        self.ffu_Cm10Cp10 = -1.
        self.ffu_Cp10Cp20 = -1.
        self.ffu_Cp20MAX  = -1.
        self.ffu_rad10 = -1.
        self.ffu_rad20 = -1.
        self.ffu_mid10 = [] # for plotting format [x,y]
        self.ffu_mid20 = [] # for plotting format [x,y]
        self.ffu_lin_unif = -1
        self.ffu_mid_linunif = [] # for plotting format [x,y]
        self.ffu_rad_linunif = -1. # for plotting format rad

        # Spatial Linearity
        self.lin_slice = -1
        self.lin_posgt = []     # for plotting format [x,y]
        self.lin_posfound = []  # for plotting format [x,y]
        self.lin_diampx = -1.   # for plotting format [x,y]
        self.lin_phantomrotdeg = 0. # phantom misalignment
        self.lin_phantomshift = []
        self.lin_sizehor = 0
        self.lin_sizever = 0
        self.lin_intshiftavg = [] # shifts
        self.lin_intshiftsdev = []
        self.lin_shiftmax = [] # geom. shifts
        self.lin_shiftmin = []
        self.lin_intdiffavg = [] # linear differentials
        self.lin_intdiffsdev =[]
        self.lin_intdiffmax = [] 
        self.lin_intdiffmin = [] 
        self.lin_nema_label = []
        self.lin_nema = []
        self.lin_nema_max = 0

        # Slice Profile
        self.sp_slice = -1
        self.sp_rois = []  # for plotting
        self.sp_mean = []  # for plotting ?
        self.sp_diamm = -1 # for plotting 
        self.sp_pins  = [] # for plotting
        self.sp_phantomrotdeg = -1
        self.sp_fwhm = None
        self.sp_fwtm = -1 # in mm
        self.sp_phantomzangledeg = -1
        self.sp_line_int = -1 # in mm
        self.sp_phantom = ""
        self.sp_method = ""
        self.sp_slicewidth_fwhm = -1 # philips PiQT report values are NOT fwhm but FWHM*tan(angle)
        self.sp_slicewidth_fwtm = -1 # philips PiQT report values are NOT fwtm but FWHM*tan(angle)
        self.sp_slicewidth_lineint = -1 # philips PiQT report values are NOT line_int but line_int*tan(angle),
        self.sp_phantomshift = 1 # philips Distance between centre of slice thickness section to the centre of image plane, but can be negative? also non-zero offset correction and scaled by (hival-loval) so it is like SW
        self.sp_phaseshift = -360. # phaseshift in degrees. Don't know if this is calculated properly

        # MTF
        self.mtf_slice = -1
        self.mtf_rois  = [] # for plotting
        self.mtf_pixelsize = []
        self.mtf_mtf50 = []
        self.mtf_integral = []

        self.resultimage = {} # container for calculated image results

class PiQT_QC:
    """
    string constants
    """
    def __init__ (self):
        self.qcversion = __version__

#----------------------------------------------------------------------
    def readDICOMtag(self,cs,key,imslice=0): # slice=2 is image 3
        value = wadwrapper_lib.readDICOMtag(key,cs.dcmInfile,imslice)
        return value


    def pix2phantommm(self,cs, pix):
        if cs.dicomMode == wadwrapper_lib.stMode2D:
            pix2mmm = cs.dcmInfile.PixelSpacing[0]
        elif cs.dicomMode == wadwrapper_lib.stMode3D:
            pix2mmm = cs.dcmInfile.info.PixelSpacing[0]
        else:
            pix2mmm = cs.dcmInfile.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[0]

        return pix*pix2mmm

    def phantommm2pix(self, cs, mm):
        if cs.dicomMode == wadwrapper_lib.stMode2D:
            pix2mmm = cs.dcmInfile.PixelSpacing[0]
        elif cs.dicomMode == wadwrapper_lib.stMode3D:
            pix2mmm = cs.dcmInfile.info.PixelSpacing[0]
        else:
            pix2mmm = cs.dcmInfile.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[0]

        return mm/pix2mmm
#----------------------------------------------------------------------
    def CoilType(self,cs,imslice):
        """
        PiQT always uses Head Coil, but for completeness sake
        Receive Coil
        """
        result = lit.stUnknown
        dicomvalue = self.readDICOMtag(cs,"0018,1250",imslice) # Receive Coil Name
        dicomvalue = str(dicomvalue).upper()
        if dicomvalue.find("BODY")>-1:
            result = lit.stCoilBody
        elif dicomvalue.find("HEAD")>-1 or dicomvalue.find("MULTI")>-1: # multi coil means both, but we only check for non_surface
            result = lit.stCoilHead
        elif dicomvalue.find("SURFACE")>-1:
            result = lit.stCoilSurface
        else:
            print("[CoilType]",dicomvalue)
        return result

    def ImageSliceNumber(self,cs,piqttest): #seqname,imagetype,slice_number,echo_time,echo_num):
        """
        Make index of slices by imagetype (M,R,I); echotime (15,30,50,100,150); echonumber (1,2,3)
        """
        result = -1
        (seqname,imagetype,slice_number,echo_num,echo_time) = piqttest
        dicomvalue = self.readDICOMtag(cs,"0018,1030",0) #"Protocol Name"],  # QA1S:MS,SE
        _seqname = str(dicomvalue).upper()
        seqid = seqname[:3] # first 3 elements QA1
        #if not _seqname[:3] in seqid: # Protocolname is renamed to WIP...
        if not seqid in _seqname:
            print("ImageSliceNumber: wrong sequence; %s != %s" %(seqid,_seqname[:3]))
            return result

        dicomfields = [
            ["2005,1011", "Image_Type"], # M,R,I
            ["2001,100a", "Slice Number"], # Philips private, alternative is slice location/slice spacing
            ["0018,0086", "Echo_No"], # 1
            # ["0018,0081", "Echo_Time"], # 50
        ]
        if cs.dicomMode == wadwrapper_lib.stMode2D:
            dimz = 1
        elif cs.dicomMode == wadwrapper_lib.stMode3D:
            dimz = len(cs.dcmInfile._datasets)
        else:
            dimz = len(cs.dcmInfile.PerFrameFunctionalGroupsSequence)
        indices = {}
        for i in range(dimz):
            val = ""
            for df in dicomfields:
                key = df[0]
                value = self.readDICOMtag(cs,key,i)
                val+= str(value).replace('"','').strip()+"_" # pydicom 2 adds "" to pydicom.valuerep.IS
            indices[val] = i
            #print(seqname,val,i)

        lookup = "" +str(imagetype)+"_"+str(slice_number)+"_"+str(echo_num)+"_" #+str(echo_time)+"_"
        if lookup in indices:
            return indices[lookup]
        else:
            return result


    def SliceProfile(self,cs_mr):
        """
        Slice profile thickness of the transverse magnetization as a function of the position
        in the selection direction. 
        The slice profile is calculated from a double ramp with a width of 1mm and a wedge. The angle
        of the ramps and the wedge with respect to the image plane is 11.31 degrees.

        Two calculations are defined:
            Modulus: Calculates the slice profile from the modulus image.
                A profile across the ramp section or differentiation of a profile along the
                wedge yields the slice profile. To increase the SNR of the calculated slice
                profile a number of profiles are averaged.
            MXY: Similar to modulus, but the averaging of profiles is performed in the real and
                imaginary image before the modulus is taken.

        The slice profile is measured as:
            FWHM: Full Width Half Maximum of slice profile.
            FWTM: Full Width at Tenth Maximum.
            Slice Integral: Integral of slice profile i.e. the energy stored in the slice.
            Phase shift: Phase difference in selection direction (only Mxy).

        This procedure consists of three steps:
            1. The in-plane rotation of the image (using the three pins) and the angle of the scan
            plane is determined with the phantom section (using the double ramp).
            2. Depending on the slice thickness the ramp or the wedge is selected:
                ramp : slice thickness >= 5 mm
                wedge: slice thickness < 5 mm
            3.The calculation method is selected, depending on the scan sequence:
                Mxy: SE and IR images
                modulus: FFE images
        NOTES:
        A. 1D linear phase correction should be applied for SE and IR scans.
        B. If 'Mxy' method fails (e.g. if no imaginary or real images are present) the 'modulus'
        method is selected automatically.

        The NEMA procedure is identical to the Philips quality procedure except that only
        'modulus' calculations are performed.

        Workflow:
            1. Determine slice thickness -> ramp (>=5mm) or wedge
            2. Determine scan sequence -> Mxy (SE,IR), Modulus (FFE)
            3. Determine in-plane rotation
            4. Determine angle of scan plane
            5. Position ROI and calculate stuff
        """
        error = True
        # 0. Setup
        if cs_mr is None or cs_mr.dcmInfile is None or cs_mr.pixeldataIn is None:
            print("[SliceProfile] Error: Not a valid PiQT struct")
            return error

        ## image sequence
        cs_mr.sp_slice = self.ImageSliceNumber(cs_mr,cs_mr.piqttest)
        if cs_mr.sp_slice <0:
            print("[SliceProfile] ERROR: Test", lit.stTestSliceProfile,"not available for given image")
            return error

        ## coiltype
        coiltype = self.CoilType(cs_mr,cs_mr.sp_slice)
        if coiltype == lit.stUnknown or coiltype == lit.stCoilSurface:
            print("[SliceProfile] ERROR: coiltype not recognized")
            return error

        # 1. Determine Ramp or Wedge
        sp_object = lit.stWedge
        if cs_mr.dicomMode == wadwrapper_lib.stMode3D:
            slicethickmm = cs_mr.dcmInfile._datasets[cs_mr.sp_slice].SliceThickness
        else: # enhanced DICOM
            try:
                slicethickmm = cs_mr.dcmInfile.PerFrameFunctionalGroupsSequence[cs_mr.sp_slice].PixelMeasuresSequence[0].SliceThickness
            except:
                slicethickmm = cs_mr.dcmInfile.PerFrameFunctionalGroupsSequence[cs_mr.sp_slice][('2005','140f')][0].SliceThickness

        if slicethickmm >=5:
            sp_object = lit.stRamp
        # AS: It seems that for newer scanners, the Ramp is always used, even if slicethickmm=2
        #   so from now on, we force Ramp
        sp_object = lit.stRamp

        # 2. Determine Mxy or Modulus
        sp_method = lit.stMxy
        if cs_mr.dicomMode == wadwrapper_lib.stMode3D:
            scanningsequence = (cs_mr.dcmInfile._datasets[cs_mr.sp_slice].ScanningSequence).upper()
        else: #enhanced DICOM
            scanningsequence = (cs_mr.dcmInfile.PerFrameFunctionalGroupsSequence[cs_mr.sp_slice][(0x2005, 0x140f)][0].ScanningSequence).upper()

        if scanningsequence.find("FFE")>-1:
            sp_method = lit.stModulus

        if sp_method == lit.stMxy:
            tup = cs_mr.piqttest
            lst = list(tup)
            lst[1] = "R"
            piqttestR = tuple(lst)
            lst[1] = "I"
            piqttestI = tuple(lst)
            cs_mr.lin_slice = self.ImageSliceNumber(cs_mr,cs_mr.piqttest)
            sp_sliceR =  self.ImageSliceNumber(cs_mr,piqttestR)
            sp_sliceI =  self.ImageSliceNumber(cs_mr,piqttestI)
            if(sp_sliceR <0 or sp_sliceI<0):
                sp_method = lit.stModulus
        print("**** Method = ",sp_method,sp_object,cs_mr.lin_slice,sp_sliceR,sp_sliceI)

        # 3. in-plane rotation from 3 pins
        """
            Just like the discs in the Spatial Linearity: 
            1. define groundtruth (3 pts)
            2. find best center (look for minimum!)
            3. rigid transformation
            4. Find angle
        """
        # theoretical positions
        wid = cs_mr.pixeldataIn[cs_mr.snr_slice].shape[0]
        hei = cs_mr.pixeldataIn[cs_mr.snr_slice].shape[1]
        xmid = (wid-1)/2.
        ymid = (hei-1)/2.
        defdistmm = 40.
        defdiamm  = 5.
        dxy = self.phantommm2pix(cs_mr,defdistmm)

        # 4. angle of scan plane rotation from the double ramp
        """
        Calculate FWHM1 and FWHM2 for ramp1 and ramp2.
        For a ramp at angle t to plane, the slice thickness S = FWHM*tan(t)
        Schneiders, Med.Phys 8(4) 1981, p516:
            Ramp1 at angle t1 to plane, ramp2 at angle t2 to plane, plane at angle phi.
            F1 is FWHM measured for ramp1, F1 is FWHM measured for ramp2.
            Define R = 2*F1/(F1+F2)*tan(t1+t2)
            then phi = arctan[1/R*(+/- sqrt(1+F2/F1*R^2)-1)]-t1; use plus sign when t1+t2<90 deg and otherwise minus

        Slice Integral: Integral of slice profile.
        Note that the wedge profile should be differentiated to get the slice profile
        Note that for Mxy first make ROIs and calculate line averages in both real and,
          imaginary, then calculate the modulus and continue to FWHM 
        """
        rois = []
        fwhm_mm = []
        fwtm_mm = []
        mean_mm = []
        line_int = []

        h0 = int(self.phantommm2pix(cs_mr,7.))
        if sp_object == lit.stRamp:
            rois.append([int(2*dxy),int(wid-4*dxy), int(2*dxy)-h0,2*h0]) # x0,wid, yo,hei
            rois.append([int(2*dxy),int(wid-4*dxy), int(3*dxy)-h0,2*h0]) # x0,wid, yo,hei
        else:
            rois.append([int(2*dxy),int(wid-4*dxy), int(hei-2.5*dxy)-h0,2*h0]) # x0,wid, yo,hei

        hwid = int(self.phantommm2pix(cs_mr,slicethickmm)/2.)
        if len(rois)>1:
            cs_mr.sp_phaseshift = []

        for r in rois:#[rois[0]]:
            if(sp_method == lit.stMxy):
                line = self.phase_shift1(cs_mr, r,sp_sliceR,sp_sliceI,sp_object)
            else:
                data = cs_mr.pixeldataIn[cs_mr.sp_slice,r[0]:(r[0]+r[1]),r[2]:(r[2]+r[3])]
                line = np.mean(data,1) # average in y direction
            if(sp_object == lit.stWedge):
                """
                The finite difference derivative suffers terribly from noise.
                The proper way is to take a Gaussian derivative, but this will have an effect
                on the FWHM measurement. Some experiments show that if the original shape is a
                block of width B, a blur with sigma < B/8. will have neglible effect on FWHM
                However in this case the block will have a width B=0...
                """
                # finite difference derivative
                line2 = np.roll(line,1)
                line = line-line2
                line[0] = line[1]
                line[-1] = line[-2]

            line_int.append(np.sum(line))
            # fit gaus to find center            
            pos = range(len(line))
            error,fit = mymath.GaussianFit(line)
            flipsign = False
            if fit[0]<0: # profile is invert, invert back and recalc!
                flipsign = True
                maxline = np.max(line)
                line = [maxline -1*l for l in line]
                line_int[-1] = np.sum(line)
                error,fit = mymath.GaussianFit(line)

            mid = int(fit[1]+.5)
            mean_mm.append(self.phantommm2pix(cs_mr,fit[1]))
            hival = np.max(line) # np.mean(line[mid-hwid:mid+hwid])
            loval = np.min(line) #0.5*(np.mean(line[0:2*hwid])+np.mean(line[len(line)-2*hwid:len(line)]))
            # correct line integral for non-zero offset?
            line_int[-1] -= len(line)*loval
            line_int[-1] /= (hival-loval) # try to calculate the slice width based on max mm for energy contained
            line_int[-1] *= self.pix2phantommm(cs_mr,1.)
            ### line_int[-1] /= hival

            hval = 0.5*(loval+hival)
            lpos = 0
            rpos = len(line)-1
            for i in range(len(line)-1):
                if line[i]<=hval and line[i+1]>=hval:
                    lpos = (hval-line[i])/(line[i+1]-line[i])+i
                    break
            for i in reversed(range(len(line)-1)):
                if line[i]>=hval and line[i+1]<=hval:
                    rpos = (hval-line[i])/(line[i+1]-line[i])+i
                    break
            fwhm_mm.append(self.phantommm2pix(cs_mr,rpos-lpos))
#            if(sp_method == lit.stMxy):
#                phaseshift = []
#                for i in range(int(lpos+(rpos-lpos)/4),int(rpos-(rpos-lpos)/4)):
#                    phaseshift.append(np.arctan(lineI[i]/lineR[i])/np.pi*180.)
#                print("phaseshift:",np.min(phaseshift),np.max(phaseshift),np.mean(phaseshift))
            tval = 0.1*(loval+hival)
            tlpos = 0
            trpos = len(line)-1
            for i in range(len(line)-1):
                if (line[i]<=tval and line[i+1]>=tval):
                    tlpos = (tval-line[i])/(line[i+1]-line[i])+i
                    break
            for i in reversed(range(len(line)-1)):
                if (line[i]>=tval and line[i+1]<=tval):
                    trpos = (tval-line[i])/(line[i+1]-line[i])+i
                    break
            fwtm_mm.append(self.phantommm2pix(cs_mr,trpos-tlpos))

            if cs_mr.verbose:
                # plot
                plt.figure()
                plt.title("fit PhaseShift Mxy "+str(r))
                gs_fit = mymath.gauss(pos, *fit)
                plt.plot(line,label='data')
                plt.plot(pos,gs_fit,label='gs fit')
                plt.plot([pos[mid-hwid],pos[mid+hwid]],[hival,hival],label='hival',linewidth=2.0)           
                plt.plot([lpos,rpos],[hval,hval],label='hval',linewidth=2.0)
                plt.plot([tlpos,trpos],[tval,tval],label='tval',linewidth=2.0)
                plt.legend()

        if len(rois)>1:
            cs_mr.sp_phaseshift = min(cs_mr.sp_phaseshift)
        if cs_mr.verbose:
            cs_mr.hasmadeplots = True

        # gui stuff
        cs_mr.sp_rois = copy.deepcopy(rois)
        cs_mr.sp_mean = copy.deepcopy(mean_mm)
        cs_mr.sp_diamm = defdiamm
        print("FWHMA!",fwhm_mm)
        if min(fwhm_mm)< 0.:
            print("Error fwhm smaller than 0.!", fwhm_mm)
            return True
        
        if len(fwhm_mm) == 1:
            cs_mr.sp_fwhm = fwhm_mm[0]
            cs_mr.sp_fwtm = fwtm_mm[0]
            cs_mr.sp_line_int = line_int[0]
            cs_mr.sp_phantomzrotdeg = 0.
        else:
            F1 = fwhm_mm[0] # order is important
            F2 = fwhm_mm[1]
            theta1 = np.pi*11.31/180.
            theta2 = theta1
            R = 2.*F1/(F1+F2)*np.tan(theta1+theta2)
            phi = (np.arctan( 1./R*(np.sqrt(1.+F2/F1*R**2)-1 )) -theta1) 

            cs_mr.sp_fwhm = [fwhm_mm[0],fwhm_mm[1]]
            cs_mr.sp_fwtm = [fwtm_mm[0],fwtm_mm[1]]
            cs_mr.sp_line_int = [line_int[0],line_int[1]]
            zrotdeg=(180*(-phi+theta1)/np.pi) # AS:  need to take negative phi value, why?
            cs_mr.sp_phantomzangledeg = zrotdeg
            cs_mr.sp_slicewidth_fwhm = fwhm_mm[0]*np.tan(phi+theta1)
            cs_mr.sp_slicewidth_fwtm = fwtm_mm[0]*np.tan(phi+theta1)
            cs_mr.sp_slicewidth_lineint = line_int[0]*np.tan(phi+theta1)
            print("[SP]",line_int,phi+theta1,np.tan(phi+theta1))
        print("FWHMB!",cs_mr.sp_fwhm)
        # slice width = inplane FWMH*tan theta
        cs_mr.sp_phantom = sp_object 
        cs_mr.sp_method = sp_method 

        #dataR = self.pixeldataIn[sp_sliceR].astype(float)
        #dataI = self.pixeldataIn[sp_sliceI].astype(float)
        #cs_mr.resultimage['phase'] = np.arctan(dataI/dataR) # phase!
        
        error = False
        return error

    def phase_shift1(self,cs_mr,r,sp_sliceR,sp_sliceI,sp_object):
        dataR = cs_mr.pixeldataIn[sp_sliceR,r[0]:(r[0]+r[1]),r[2]:(r[2]+r[3])]
        lineR = np.mean(dataR,1) # average in y direction
        dataI = cs_mr.pixeldataIn[sp_sliceI,r[0]:(r[0]+r[1]),r[2]:(r[2]+r[3])]
        lineI = np.mean(dataI,1) # average in y direction
        dataM = cs_mr.pixeldataIn[cs_mr.sp_slice,r[0]:(r[0]+r[1]),r[2]:(r[2]+r[3])]
        lineM = np.mean(dataM,1) # average in y direction
        line = []
        phaseshift = [ ]
        """
        wid = cs_mr.pixeldataIn[cs_mr.sp_slice].shape[0]
        hei = cs_mr.pixeldataIn[cs_mr.sp_slice].shape[1]
        for y in range(hei):
            for x in range(wid):
                cs_mr.pixeldataIn[cs_mr.sp_slice,x,y] = 180./np.pi*np.arctan2(cs_mr.pixeldataIn[sp_sliceI,x,y],cs_mr.pixeldataIn[sp_sliceR,x,y])
        """
        for rr,ii,mm in zip(lineR,lineI,lineM):
            line.append(np.sqrt(rr**2+ii**2))
            phaseshift.append(np.arctan2(ii,rr)/np.pi*180.)

        # blur line, find pos max, average phaseshift over pos_max +/- 3
        line_sigma = 2.8 # dimensionless
        sp_phaseshift = []
        for oneline in [lineR,lineI]:
            # find locations of extremes close to center
            if(sp_object == lit.stWedge):
                blur_line = scind.filters.gaussian_filter1d(oneline, line_sigma, order=1)
            else:
                blur_line = scind.filters.gaussian_filter1d(oneline, line_sigma, order=0)
            line_maxid = np.unravel_index(blur_line.argmax(), blur_line.shape)[0]
            line_minid = np.unravel_index(blur_line.argmin(), blur_line.shape)[0]
            if abs(line_minid-blur_line.shape[0]/2)<abs(line_maxid-blur_line.shape[0]/2): # min or max close to center
                line_maxid = line_minid

            if 3<line_maxid<len(oneline)-3:
                blur_phaseshift = scind.filters.gaussian_filter1d(phaseshift, line_sigma, order=0)
                sp_phaseshift.append(blur_phaseshift[line_maxid])
                print("phaseshift:",sp_phaseshift[-1])
            else:
                print("phaseshift:","dunno")
                sp_phaseshift.append(-360.)
        cs_mr.sp_phaseshift.append(sp_phaseshift[0]-sp_phaseshift[1])
        print("dphaseshift:",cs_mr.sp_phaseshift[-1])

        if cs_mr.verbose:
            plt.figure()
            plt.title("data PhaseShift Mxy "+str(r))
            plt.plot(phaseshift,lineR,label='real')
            plt.plot(phaseshift,lineI,label='imag')
            plt.plot(phaseshift,line,label='mag',linewidth=2.0)
            plt.legend()
            plt.figure()
            plt.plot(phaseshift[35:60])
        return line

    def phase_shift(self,cs_mr,r,sp_sliceR,sp_sliceI,sp_object):
        dataR = cs_mr.pixeldataIn[sp_sliceR,r[0]:(r[0]+r[1]),r[2]:(r[2]+r[3])]
        lineR = np.mean(dataR,1) # average in y direction
        dataI = cs_mr.pixeldataIn[sp_sliceI,r[0]:(r[0]+r[1]),r[2]:(r[2]+r[3])]
        lineI = np.mean(dataI,1) # average in y direction
        dataM = cs_mr.pixeldataIn[cs_mr.sp_slice,r[0]:(r[0]+r[1]),r[2]:(r[2]+r[3])]
        lineM = np.mean(dataM,1) # average in y direction
        line = []
        phaseshift = [ ]
        for rr,ii,mm in zip(lineR,lineI,lineM):
            line.append(np.sqrt(rr**2+ii**2))
            phaseshift.append(np.arctan2(ii,rr)/np.pi*180.)

        # blur line, find pos max, average phaseshift over pos_max +/- 3
        line_sigma = 2.8 # dimensionless
        if(sp_object == lit.stWedge):
            blur_line = scind.filters.gaussian_filter1d(line, line_sigma, order=1)
        else:
            blur_line = scind.filters.gaussian_filter1d(line, line_sigma, order=0)
        line_maxid = np.unravel_index(blur_line.argmax(), blur_line.shape)[0]
        line_minid = np.unravel_index(blur_line.argmin(), blur_line.shape)[0]
        if abs(line_minid-blur_line.shape[0]/2)<abs(line_maxid-blur_line.shape[0]/2): # min or max close to center
            line_maxid = line_minid
        if 3<line_maxid<len(line)-3:
            blur_phaseshift = scind.filters.gaussian_filter1d(phaseshift, line_sigma, order=0)
            cs_mr.sp_phaseshift.append(blur_phaseshift[line_maxid])
            print("phaseshift:",cs_mr.sp_phaseshift[-1])
        else:
            print("phaseshift:","dunno")
            cs_mr.sp_phaseshift.append(-360.)
        if cs_mr.verbose:
            plt.figure()
            plt.title("data PhaseShift Mxy "+str(r))
            plt.plot(lineR,label='real')
            plt.plot(lineI,label='imag')
            plt.plot(line,label='mag',linewidth=2.0)
            plt.legend()
        return line

    def MTF(self,cs_mr):
        """
        The Philips phantom for measuring spatial resolution is the 'square hole' section of the
        head phantom. The procedure for evaluation of an image of this section is as follows:
            For the measurement and the preparation direction the Edge Response Function
                (ERF) is determined from the 'image' of the edge of the square hole.
                Differentiation of the ERFs yield the Line Spread Functions (LSF) in the measure-
                ment and preparation directions.
            The MTFs in measurement and preparation directions are obtained by taking the
                Fourier Transforms of the LSFs.
        To characterize the LSFs and MTFs the following parameters are calculated for both the
        measurement and preparation direction:
            pixel size: Width between the two first zeros of the LSF.
            MTF_50: Full Width at Half Maximum, FWHM, expressed in line pairs per pixel.
                For an ideal system this value is 0.5.
            Integral: Integral of MTF between 0 and 0.5

        """
        """
        Workflow:
            1. Find phantom rotation
            2. Make ROIs and Calc EdgeSpreadFunction
            3. Do FFT
            4. Calc stuff
        """
        error = True
        # 0. Setup
        if cs_mr is None or cs_mr.dcmInfile is None or cs_mr.pixeldataIn is None:
            print("[MTF] Error: Not a valid PiQT struct")
            return error

        ## image sequence
        cs_mr.mtf_slice = self.ImageSliceNumber(cs_mr,cs_mr.piqttest)
        if cs_mr.mtf_slice <0:
            print("[MTF] ERROR: Test", lit.stTestMTF,"not available for given image")
            return error

        """
        1. Phantomrotation:
        1.a Make lines between edge and first half of phantom
        1.b Calculate number of phantom pixels in line (sum over line - N*bk)/sig
        1.c Fit line to slopes
        1.d Best fit of four sides gives rotation
        """
        cs_mr.mtf_rois = []
        # theoretical positions
        wid = cs_mr.pixeldataIn[cs_mr.mtf_slice].shape[0]
        hei = cs_mr.pixeldataIn[cs_mr.mtf_slice].shape[1]
        xmid   = (wid-1)/2.
        ymid   = (hei-1)/2.
        diampx = wid/3
        radsig = 10
        cs_mr.mtf_rois.append([10,2*radsig, 10,2*radsig]) # bk  # x0,wid, y0,hei
        cs_mr.mtf_rois.append([int(xmid-radsig),  2*radsig,      int(hei/8+diampx/2)-radsig,2*radsig]) # signal  # x0,wid, y0,hei
        cs_mr.mtf_rois.append([int(xmid-diampx/2),int(diampx/2), int(hei/8),int(diampx)])   # VOI left half # x0,wid, y0,hei
        cs_mr.mtf_rois.append([int(xmid),         int(diampx/2), int(hei/8),int(diampx)])   # VOI right half # x0,wid, y0,hei
        cs_mr.mtf_rois.append([int(xmid-diampx/2),int(diampx), int(hei/8),int(diampx/2)])   # VOI upper half # x0,wid, y0,hei
        cs_mr.mtf_rois.append([int(xmid-diampx/2),int(diampx), int(hei/8+diampx/2),int(diampx/2)])   # VOI lower half # x0,wid, y0,hei
        # BK calc
        r = cs_mr.mtf_rois[0]
        data = cs_mr.pixeldataIn[cs_mr.mtf_slice,r[0]:(r[0]+r[1]),r[2]:(r[2]+r[3])]
        bk_mean  = np.mean(data)
        bk_stdev = np.std(data)
        # Sig calc
        r = cs_mr.mtf_rois[1]
        data = cs_mr.pixeldataIn[cs_mr.mtf_slice,r[0]:(r[0]+r[1]),r[2]:(r[2]+r[3])]
        sig_mean  = np.mean(data)
        sig_stdev = np.std(data)

        if cs_mr.verbose:
            #plt.figure()
            cs_mr.hasmadeplots = True
        phantomangle = 0.
        best_r = 0.
        for r in cs_mr.mtf_rois[2:]:
            pline = []
            if r[1]>r[3]:
                for y in range(r[2],r[2]+r[3]):
                    data = cs_mr.pixeldataIn[cs_mr.mtf_slice,r[0]:(r[0]+r[1]),y]
                    pline.append((np.sum(data)-len(data)*bk_mean)/(sig_mean-bk_mean))
            else:
                for x in range(r[0],r[0]+r[1]):
                    data = cs_mr.pixeldataIn[cs_mr.mtf_slice,x,r[2]:(r[2]+r[3])]
                    pline.append((np.sum(data)-len(data)*bk_mean)/(sig_mean-bk_mean))
            if pline[0]>pline[-1]:
                pline = list(reversed(pline))
            fitline = []
            maxV = np.mean(pline[-5:-1])
            minV = np.mean(pline[0:5])
            ranV = (maxV-minV)
            for y in pline:
                if y > minV+ranV/8. and y<maxV-ranV/8.:
                    fitline.append(y)

            if cs_mr.verbose:
                plt.figure()
                plt.title('(signal-bk)/signal_mean')
                plt.plot(pline,label='(signal-bk)/signal_mean')
                #plt.plot(fitline,label= 'fit')
                #plt.legend()
                #plt.show()
            if len(fitline)<2:
                raise ValueError("[MTF] Cannot find EdgeObject! Rotated phantom?")
            error,coef = mymath.LinearFit(fitline)
            if coef[2]>best_r:
                best_r = coef[2]
                phantomangle = np.arctan(coef[0])
#            print(len(fitline),np.arctan(coef[0])/np.pi*180.,coef[2:])
#        phantomangle = (90-11.3)/180.*np.pi
        print("phantomangledeg",phantomangle/np.pi*180-90.)

        # 2. make 2 MTF ROIs and calculated presampled MTFs
        radsig = 5
        cs_mr.mtf_rois.append([int(xmid),2*radsig,   cs_mr.mtf_rois[1][2]+cs_mr.mtf_rois[1][3],int(diampx/2)]) # signal  # x0,wid, y0,hei
        cs_mr.mtf_rois.append([cs_mr.mtf_rois[1][0]+cs_mr.mtf_rois[1][1],int(diampx/2), cs_mr.mtf_rois[1][2],2*radsig]) # signal  # x0,wid, y0,hei

        lineH = []
        lineV = []
        cosval = np.cos(phantomangle)
        doScaleMTF = True
        for r in cs_mr.mtf_rois[-2:]:
            if r[1]>r[3]:
                for y in range(r[2],r[2]+r[3]):
                    data = cs_mr.pixeldataIn[cs_mr.mtf_slice,r[0]:(r[0]+r[1]),y].astype(float)
                    if doScaleMTF:
                        hival = np.mean(data[0:10])
                        loval = np.mean(data[-11:-1])
                        if loval > hival:
                            swapval = loval
                            loval = hival
                            hival = swapval
                        for i in range(len(data)):
                            data[i] = (data[i]-loval)/(hival-loval)
                    for da in enumerate(data):
                        lineH.append( (da[0]-y*cosval,da[1]) )
            else:
                for x in range(r[0],r[0]+r[1]):
                    data = cs_mr.pixeldataIn[cs_mr.mtf_slice,x,r[2]:(r[2]+r[3])].astype(float)
                    if doScaleMTF:
                        hival = np.mean(data[0:10])
                        loval = np.mean(data[-11:-1])
                        if loval > hival:
                            swapval = loval
                            loval = hival
                            hival = swapval
                        for i in range(len(data)):
                            data[i] = (data[i]-loval)/(hival-loval)
                    for da in enumerate(data):
                        lineV.append( (da[0]+x*cosval,da[1]) )
        lineH = sorted(lineH)
        lineV = sorted(lineV)
        # 3. 1D FFT
        # interpolate
        cs_mr.mtf_pixelsize = [0.,0.]
        cs_mr.mtf_pixelsize[0] = self.MTFcalc(cs_mr,lineH,cosval,cs_mr.verbose)
        cs_mr.mtf_pixelsize[1] = self.MTFcalc(cs_mr,lineV,cosval,cs_mr.verbose)
##        self.EdgeDetection(self.pixeldataIn[cs_mr.mtf_slice])
        error = False
        return error

    def MTFcalc(self,cs_mr,linedat,stepsize_in,verbose):
        from scipy.interpolate import interp1d
        stepsize = stepsize_in

        # interpolate
        xH = np.array([row[0] for row in linedat])
        yH = np.array([row[1] for row in linedat])
        hfun = interp1d(xH,yH, kind='linear')
        hpos = np.arange(xH[0],xH[-1],stepsize)
        esf = hfun(hpos)
        esfMax = max(esf)
        esfMin = min(esf)
        esf=(esf-esfMin)/(esfMax-esfMin)
        doCenter = True
        if doCenter:
            sigma = 2.8 # dimensionless
            lsf = scind.filters.gaussian_filter1d(esf, sigma, order=1)
            lsfmaxid = np.unravel_index(lsf.argmin(), lsf.shape)[0]
            nums = min(len(lsf)-lsfmaxid,lsfmaxid+1)
            esf = esf[lsfmaxid-nums+1:lsfmaxid+nums]
            hpos = hpos[lsfmaxid-nums+1:lsfmaxid+nums]
        if verbose:
            plt.figure()
            plt.title("esf")
            #plt.plot(xH,yH,'b.')
            plt.plot(hpos,esf,'r.')

        # differentiate
        useGaussian = True
        if not useGaussian:
            line2 = np.roll(esf,1)
            lsf = (esf-line2)
            lsf[0] = lsf[1]
            lsf[-1] = lsf[-2]
        else:
            sigma = 0.8 # dimensionless
            lsf = scind.filters.gaussian_filter1d(esf, sigma, order=1)

        # normalize LSF
        lsf  = lsf/np.sum(lsf)
        lsfmaxid = np.unravel_index(lsf.argmax(), lsf.shape)[0]
        lsfminid = np.unravel_index(lsf.argmin(), lsf.shape)[0]

        # find first left zero crossing
        lpos = None
        for k in reversed(range(1,lsfmaxid)):
            if lsf[k] >=0. and lsf[k-1] <=0.:
                lpos = hpos[k]+(0.-lsf[k])/(lsf[k-1]-lsf[k])*(hpos[k-1]-hpos[k])
                break
        if lpos is None:
            print("ERROR: lpos is None")
            return -1.

        # find second left zero crossing
        l2pos = None
        for k in reversed(range(1,int((lpos-hpos[0])/(hpos[1]-hpos[0]))+1)):
            if lsf[k] <=0. and lsf[k-1] >=0.:
                l2pos = hpos[k]+(0.-lsf[k])/(lsf[k-1]-lsf[k])*(hpos[k-1]-hpos[k])
                break
        if l2pos is None:
            print("ERROR: l2pos is None")
            return -1.

        # find third left zero crossing
        l3pos = None
        for k in reversed(range(1,int((l2pos-hpos[0])/(hpos[1]-hpos[0]))+1)):
            if lsf[k] >=0. and lsf[k-1] <=0.:
                l3pos = hpos[k]+(0.-lsf[k])/(lsf[k-1]-lsf[k])*(hpos[k-1]-hpos[k])
                break
        if l3pos is None:
            print("ERROR: l3pos is None")
            return -1.

        # find first right zero crossing
        rpos = None
        for k in range(lsfmaxid,len(lsf)-1):
            if lsf[k] >=0. and lsf[k+1] <=0.:
                rpos = hpos[k]+(0.-lsf[k])/(lsf[k+1]-lsf[k])*(hpos[k+1]-hpos[k])
                break
        if rpos is None:
            print("ERROR: right pos is None")
            return -1.

        # find second right zero crossing
        for k in range(int((rpos-hpos[0])/(hpos[1]-hpos[0]))+1,len(lsf)-1):
            if lsf[k] >=0. and lsf[k+1] <=0.:
                r2pos = hpos[k]+(0.-lsf[k])/(lsf[k+1]-lsf[k])*(hpos[k+1]-hpos[k])
                break

        pixsize = self.pix2phantommm(cs_mr,lpos-l3pos)/2.
        #pixsize = 2./3.*self.pix2phantommm(cs_mr,abs(hpos[lsfmaxid]-hpos[lsfminid]))
        if verbose:
            print("pixsize=",pixsize,self.pix2phantommm(cs_mr,rpos-lpos),self.pix2phantommm(cs_mr,r2pos-rpos),self.pix2phantommm(cs_mr,l2pos-lpos),self.pix2phantommm(cs_mr,l3pos-l2pos),self.pix2phantommm(cs_mr,1.))
            plt.figure()
            plt.title("lsf")
            plt.plot(hpos,lsf,label='lsf')
            #plt.xlim((l3pos,r2pos))
            plt.plot([l3pos,l2pos,lpos,rpos,r2pos],[0.,0.,0.,0.,0.],'r.',label='zero')
            plt.plot([hpos[lsfmaxid],hpos[lsfminid]],[0.,0],'b.',label='zero2')

        # take FFT
        from numpy import fft as fftpack
        mtf = np.abs(fftpack.fft(lsf))
        # frequency axis
        freq = fftpack.fftfreq(len(lsf), d=self.pix2phantommm(cs_mr,stepsize))
        mtf = mtf[0:int(len(mtf)/2)-1]
        freq = freq[0:int(len(freq)/2)-1]

        fnyq = .5/self.pix2phantommm(cs_mr,1.)
        if verbose:
            plt.figure()
            plt.title("MTF")
            plt.plot(freq,mtf)
            plt.xlim(0,fnyq)

        integral50 = mymath.AreaUnderCurve(freq,mtf,0.5,normalized=False)
        for k in range(0,len(mtf)-1):
            if mtf[k] >=0.5 and mtf[k+1] <=0.5:
                mtf50 = freq[k]+(0.5-mtf[k])/(mtf[k+1]-mtf[k])*(freq[k+1]-freq[k])
                break
        print("mtf50/integral50",mtf50,integral50)

        return pixsize


    def SNR_NEMA2(self, cs_mr):
        """
        SNR calculated using 2 images to determine noise. This requires both images are
        exactly the same, which is not the case for PIQT sets, so do NOT use.

        The SNR is defined as: SNR = sqrt(2)*R/sd(B) with,
        R: Mean pixel value of ROI at reference position.
        The reference position depends on the coil type.
        sd(B): Standard deviation of pixel values in same ROI but calculated
               in the difference image.

        Because the snr_means are also used for Artifacts, do not overwrite them
        but use same ordering as NEMA1.
        
        Workflow:
            1. determine sequence (this determines which imslice to use)
            2. determine body_head coil or surface coils
            3. define ROI
            4. calculate roi avgs and stdevs
            5. return all in structure
        """
        error = True
        if cs_mr is None or cs_mr.dcmInfile is None or cs_mr.pixeldataIn is None:
            print("[SNR] Error: Not a valid PiQT struct")
            return error

        # 1. image sequence
        cs_mr.snr_slice = self.ImageSliceNumber(cs_mr,cs_mr.piqttest)
        if cs_mr.snr_slice <0:
            (seqname,imagetype,slice_number,echo_num,echo_time) = cs_mr.piqttest
            print("[SNR] ERROR: Test", lit.stTestUniformity,"not available for given image",imagetype,echo_num)
            return error

        # 1.2. second image look for next echo
        piqttest2 = [ c+1 if i == 3 else c for i,c in enumerate(cs_mr.piqttest) ]
        cs_mr.snr_slice2 = self.ImageSliceNumber(cs_mr,piqttest2)
        if cs_mr.snr_slice2 <0:
            (seqname2,imagetype2,slice_number2,echo_num2,echo_time2) = piqttest2
            print("[SNR] ERROR: Test", lit.stTestUniformity,"using NEMA2; no 2nd image available", imagetype2, echo_num2)
            return error

        # 2. coiltype
        coiltype = self.CoilType(cs_mr,cs_mr.snr_slice)
        if coiltype == lit.stUnknown or coiltype == lit.stCoilSurface:
            print("[SNR] ERROR: coiltype not recognized")
            return error

        # 3. SNR
        cs_mr.snr_means  = []
        cs_mr.snr_stdevs = []

        wid = cs_mr.pixeldataIn.shape[1]
        hei = cs_mr.pixeldataIn.shape[2]

        roiwidth  = 32 #32
        roiheight = 32 #32
        roidx     = 4 #4
        roidy     = 4 #4
        cs_mr.snr_rois = [] # format: x0,wid, y0,hei
        cs_mr.snr_rois.append([int((wid-roiwidth)/2),roiwidth, int((hei-roiheight)/2), roiheight])
        cs_mr.snr_rois.append([roidx,roiwidth, roidy,roiheight])
        cs_mr.snr_rois.append([roidx,roiwidth, hei-roiheight-roidy,roiheight])
        cs_mr.snr_rois.append([wid-roiwidth-roidx,roiwidth, hei-roiheight-roidy,roiheight])
        cs_mr.snr_rois.append([wid-roiwidth-roidx,roiwidth, roidy,roiheight])

        for r in cs_mr.snr_rois:
            data = cs_mr.pixeldataIn[cs_mr.snr_slice,r[0]:(r[0]+r[1]),r[2]:(r[2]+r[3])]
            cs_mr.snr_means.append(np.mean(data))
            cs_mr.snr_stdevs.append(np.std(data))
        for i in range(2,len(cs_mr.snr_rois)):
            if cs_mr.snr_stdevs[i]<cs_mr.snr_stdevs[1]:
                cs_mr.snr_stdevs[1] = cs_mr.snr_stdevs[i]
                cs_mr.snr_means[1] = cs_mr.snr_means[i]
                cs_mr.snr_rois[1] = cs_mr.snr_rois[i]
        cs_mr.snr_stdevs = cs_mr.snr_stdevs[0:2]
        cs_mr.snr_means = cs_mr.snr_means[0:2]
        cs_mr.snr_rois = cs_mr.snr_rois[0:2]
        
        cs_mr.snr_bkgrnd = cs_mr.snr_means[1] # for Artifacts and FFU

        # add difference image
        r = cs_mr.snr_rois[0]
        mean = cs_mr.pixeldataIn[cs_mr.snr_slice][r[0]:(r[0]+r[1]),r[2]:(r[2]+r[3])]

        m1 = np.mean(cs_mr.pixeldataIn[cs_mr.snr_slice,r[0]:(r[0]+r[1]),r[2]:(r[2]+r[3])])
        m2 = np.mean(cs_mr.pixeldataIn[cs_mr.snr_slice2,r[0]:(r[0]+r[1]),r[2]:(r[2]+r[3])])

        print("NEMA2: m1/m2 = ", m1/m2)
        diff = (cs_mr.pixeldataIn[cs_mr.snr_slice]-cs_mr.pixeldataIn[cs_mr.snr_slice2])[r[0]:(r[0]+r[1]),r[2]:(r[2]+r[3])]

        cs_mr.snr_means[0] = np.mean(mean)
        cs_mr.snr_stdevs[0] = np.std(mean)
        cs_mr.snr_means[1] = np.mean(diff)
        cs_mr.snr_stdevs[1] = np.std(diff)
        if not 'SNR_DIFF' in cs_mr.resultimage.keys():
            cs_mr.resultimage['SNR_DIFF'] = cs_mr.pixeldataIn[cs_mr.snr_slice]-cs_mr.pixeldataIn[cs_mr.snr_slice2]
            cs_mr.resultimage['SNR_SIGNAL'] = cs_mr.pixeldataIn[cs_mr.snr_slice]
        
        # report values
        cs_mr.snr_SNC = cs_mr.snr_means[0]/cs_mr.snr_stdevs[0]
        cs_mr.snr_SNB = np.sqrt(2.)*cs_mr.snr_means[0]/cs_mr.snr_stdevs[1]
        cs_mr.snr_BsdB = cs_mr.snr_means[1]/cs_mr.snr_stdevs[1]

        error = False
        return error

    def SNR_NEMA1(self, cs_mr):
        """
        SNR calculated using 1 image to calculated Signal and Noise

        The SNR is defined as: SNR = 0.655*R/sd(B) with,
        R: Mean pixel value of ROI at reference position.
        The reference position depends on the coil type.
        sd(B): Standard deviation of pixel values of ROI in a ghost-free part of the
        background of the image.
        0.655: Correction factor for backfolding of noise in background.

        Workflow:
            1. determine sequence (this determines which imslice to use)
            2. determine body_head coil or surface coils
            3. define ROI
            4. calculate roi avgs and stdevs
            5. return all in structure
        """
        error = True
        if cs_mr is None or cs_mr.dcmInfile is None or cs_mr.pixeldataIn is None:
            print("[SNR] Error: Not a valid PiQT struct")
            return error

        # 1. image sequence
        cs_mr.snr_slice = self.ImageSliceNumber(cs_mr,cs_mr.piqttest)
        if cs_mr.snr_slice <0:
            (seqname,imagetype,slice_number,echo_num,echo_time) = cs_mr.piqttest
            print("[SNR] ERROR: Test", lit.stTestUniformity,"not available for given image",imagetype,echo_num)
            return error

        # 2. coiltype
        coiltype = self.CoilType(cs_mr,cs_mr.snr_slice)
        if coiltype == lit.stUnknown or coiltype == lit.stCoilSurface:
            print("[SNR] ERROR: coiltype not recognized")
            return error

        # 3. SNR
        cs_mr.snr_means  = []
        cs_mr.snr_stdevs = []

        wid = cs_mr.pixeldataIn.shape[1]
        hei = cs_mr.pixeldataIn.shape[2]

        roiwidth  = 48 #32
        roiheight = 48 #32
        roiwidthsd  = 32 #32
        roiheightsd = 32 #32
        roidx     = 4 #4
        roidy     = 4 #4
        cs_mr.snr_rois = [] # format: x0,wid, y0,hei
        cs_mr.snr_rois.append([int((wid-roiwidth)/2),roiwidth, int((hei-roiheight)/2), roiheight])
        # try to move the noise ROI to an artefact-free part of the background
        for roidx in [4,6,8,10,12,14]:
            for roidy in [4,6,8,10,12,14]:
                for roiwidthsd in [8,16,32,64]:
                    roiheightsd = int(32*32./roiwidthsd)
                    cs_mr.snr_rois.append([roidx,roiwidthsd, roidy,roiheightsd])
                    cs_mr.snr_rois.append([roidx,roiwidthsd, hei-roiheightsd-roidy,roiheightsd])
                    cs_mr.snr_rois.append([wid-roiwidthsd-roidx,roiwidthsd, hei-roiheightsd-roidy,roiheightsd])
                    cs_mr.snr_rois.append([wid-roiwidthsd-roidx,roiwidthsd, roidy,roiheightsd])

        r = cs_mr.snr_rois[0]
        #print(np.mean(cs_mr.pixeldataIn[cs_mr.snr_slice,r[0]:(r[0]+r[1]),r[2]:(r[2]+r[3])]), np.mean(cs_mr.pixeldataIn[cs_mr.snr_slice+1,r[0]:(r[0]+r[1]),r[2]:(r[2]+r[3])]))
        for r in cs_mr.snr_rois:
            data = cs_mr.pixeldataIn[cs_mr.snr_slice,r[0]:(r[0]+r[1]),r[2]:(r[2]+r[3])]
            cs_mr.snr_means.append(np.mean(data))
            cs_mr.snr_stdevs.append(np.std(data))
        for i in range(2,len(cs_mr.snr_rois)):
            if cs_mr.snr_stdevs[i]<cs_mr.snr_stdevs[1]:
                cs_mr.snr_stdevs[1] = cs_mr.snr_stdevs[i]
                cs_mr.snr_means[1] = cs_mr.snr_means[i]
                cs_mr.snr_rois[1] = cs_mr.snr_rois[i]
        cs_mr.snr_stdevs = cs_mr.snr_stdevs[0:2]
        cs_mr.snr_means = cs_mr.snr_means[0:2]
        cs_mr.snr_rois = cs_mr.snr_rois[0:2]

        cs_mr.snr_bkgrnd = cs_mr.snr_means[1] # for Artifacts and FFU

        # report values
        cs_mr.snr_SNC = cs_mr.snr_means[0]/cs_mr.snr_stdevs[0]
        cs_mr.snr_SNB = 0.655*cs_mr.snr_means[0]/cs_mr.snr_stdevs[1]
        cs_mr.snr_BsdB = cs_mr.snr_means[1]/cs_mr.snr_stdevs[1]

        error = False
        return error

    def SNR(self,cs_mr):
        """
        Test if 2 images are available for SNR. If so, use SNR_NEMA2, else use SNR_NEMA1
        """
        error = True
        if cs_mr is None or cs_mr.dcmInfile is None or cs_mr.pixeldataIn is None:
            print("[SNR] Error: Not a valid PiQT struct")
            return error

        # 1. image sequence
        cs_mr.snr_slice = self.ImageSliceNumber(cs_mr,cs_mr.piqttest)
        (seqname,imagetype,slice_number,echo_num,echo_time) = cs_mr.piqttest
        if cs_mr.snr_slice <0:
            print("[SNR] ERROR: Test", lit.stTestUniformity,"not available for given image",imagetype,echo_num)
            return error

        # Found 1 image, see if another one available
        piqttest2 = [ c+1 if i == 3 else c for i,c in enumerate(cs_mr.piqttest) ]
        (seqname2,imagetype2,slice_number2,echo_num2,echo_time2) = piqttest2
        cs_mr.snr_slice2 = self.ImageSliceNumber(cs_mr,piqttest2)
        # disable SNR_NEMA2 as echo2 and echo 1 are not (always) identical
        cs_mr.snr_slice2 = -1
        if cs_mr.snr_slice2 <0:
            print("[SNR] Test", lit.stTestUniformity,"no 2nd image available; will use NEMA method 1",imagetype,echo_num)
            return self.SNR_NEMA1(cs_mr)
        else:
            print("[SNR] Test", lit.stTestUniformity,"2 images available; will use NEMA method 2",imagetype2,echo_num, echo_num2)
            return self.SNR_NEMA2(cs_mr)


    def _movingaverage(self, data, ksize):
        # apply a moving average to of window width ksize
        kernel = np.ones((ksize,ksize),dtype=float)
        kernel *= 1./(ksize*ksize)

        return scind.convolve(data, kernel, mode='reflect')

    def _lowpassfilter(self,data):
        # NEMA MS 3-2003
        # apply a 9 points low-pass filter
        kernel = np.array([
            [1.,2.,1.],
            [2.,4.,2.],
            [1.,2.,1.]
        ])
        kernel *= 1./16.

        return scind.convolve(data, kernel, mode='reflect')

    def ArtifactLevel(self,cs_mr):
        """
        This procedure is very similar to the NEMA Ghosting Level.

        The artifact level is defined as: artefact level = (G-B)*100/2R % with,
        G: Maximum mean value of ROI of 3*3 pixels in background of image. The edges of the image are masked.
        B: Mean pixel value of ROI in ghost-free part of background.
        R: Mean pixel value of ROI at reference position. The reference position depends on the coil type.

        Workflow:
            1. determine sequence (this determines which imslice to use)
            2. determine body_head coil or surface coils
            3. define ROIs
            4. calculate roi avgs and stdevs and in smoothed slice
            5. return all in structure
        """
        cs_mr.snr_slice = self.ImageSliceNumber(cs_mr,cs_mr.piqttest)
        error = self.SNR(cs_mr)
        if error:
            print("[Artifact] Error: Not a valid PiQT struct")
            return error
        signal = cs_mr.snr_means[0]
        bkgrnd = cs_mr.snr_bkgrnd

        # 3. Artifact
        edge = 6 # 6

        # 3.1 moving average of image
        data = self._lowpassfilter(cs_mr.pixeldataIn[cs_mr.snr_slice].astype(float))
        #data = self._movingaverage(cs_mr.pixeldataIn[cs_mr.snr_slice].astype(float),3)

        # 3.2 stay away from edges
        wid = data.shape[0]
        data = data[edge:wid-edge,edge:wid-edge]

        # 3.3 select data outside circle only and stay away from circle edge
        wid = data.shape[0]
        x,y = np.indices((wid, wid))
        mid = wid/2
        dx = 0
        dy = 0
        rad = 110 #115 # dit is een gok. Minimaal 103+ksize, maar om ghosting buiten te sluiten meer. 128 is max
        mask = ((x-mid+dy)**2 + (y-mid+dy)**2 ) > rad**2

        cs_mr.artefact_max = np.max(data[mask])
        #rad = 85 # dit is een gok. Minimaal 103+ksize, maar om ghosting buiten te sluiten meer. 128 is max
        #mask = ((x-mid)**2 + (y-mid)**2 ) < rad**2
        #signal = np.average(data[mask])

        cs_mr.artefact_roi = [mid+edge+dx,mid+edge+dy,rad]
        # report value
        cs_mr.artefact_ArtLevel = 100.*(cs_mr.artefact_max-bkgrnd)/signal/2.
        #print("[ArtifactLevel] max",cs_mr.artefact_max)
        #plt.figure()
        #data[mask] = 1500
        #plt.imshow(data.transpose())
        #cs_mr.hasmadeplots = True
        #for y in range(0,data.shape[0]):
        #    for x in range(0,data.shape[0]):
        #        cs_mr.pixeldataIn[cs_mr.snr_slice,edge+x,edge+y] = data[x,y]

        error = False
        return error

    def FloodFieldUniformity(self,cs_mr):
        """
        Make some contour plots for visual inspection and calculate different values 
         to characterize the uniformity. 
        
        The specific values have the following meanings: 
         C-20/C-10 means: percentage of pixels with values between C-20% and C-10%, 
           taking only into account the pixels with values larger than T

        A contour ('grey scale') plot of the image is created as follows:

            1. The mean pixel values C (centre) and B (background),
                of ROIs at a reference position R are calculated.
                The reference position depends on the coil type.
            2. A grey value is assigned to every pixel using the following algorithm:
             For N=2: (Head and Body coils)
                T           < pixel value < C-N*S       black  T/C-20
                C-N*S       < pixel value < C-(N-1)*S   grey 1 C-20/C-10
                C-(N-1)*S   < pixel value < C+(N-1)*S   grey 2 C-10/C+10
                C+(N-1)*S   < pixel value < C+N*S       grey 3 C+10/C+20
                C+N*S       < pixel value < MAX         white  C+20/MAX
             For N=3: (Surface coils)
                T           < pixel value < C-N*S       black
                C-N*S       < pixel value < C-(N-1)*S   grey 1
                C-(N-1)*S   < pixel value < C-(N-2)*S   grey 2
                C-(N-2)*S   < pixel value < C+(N-2)*S   grey 3
                C+(N-2)*S   < pixel value < C+(N-1)*S   grey 4
                C+(N-1)*S   < pixel value < C+N*S       grey 5
                C+N*S       < pixel value < MAX         white
             with:
                 T: threshold value which is approximately 10*B
                 MAX: maximum pixel value
                 N: number of steps
                 S: step size = percentage*(C-B) 10% for Head and Body coils, 20% for Surface coils
                 2*N+1:number of contours, i.e. grey values

        Histogram calculations:

            1. A ROI (region of interest) for histogram calculation is defined:
                size and shape depends on the coil type.
            2. The number of pixels within the histogram ROI which have a pixel value larger than
                T, i.e. pixels which are not black, is determined: Ntot.
            3. For every grey value the percentage ratio is calculated:
                percentage ratio = <number of pixels with a certain grey value> / Ntot.
            4. The percentage ratios are used to specify the floodfield uniformity.


        rad_10%: Radius of the largest circle which fits in the C-10/C+10 area. grey2
        rad_20%: Radius of the largest circle which fits in the C-20/C+20 area. grey1+grey2+grey3

        Workflow:
            1. Find C and B
            2. Smooth image slice
            3. Calculate contourimage
            4. Calculate number of pixels in each "contour"
            5. Calculate rad_10% and rad_20% by Euclidan distance transform
            6. Return all in structure
        """
        # Step 1
        error = self.ArtifactLevel(cs_mr)
        if error == True:
            print("[Artifact] Error: Not a valid PiQT struct")
            return error
        C = cs_mr.snr_means[0]
        B = cs_mr.snr_bkgrnd

        # Step 2
        data = self._lowpassfilter(cs_mr.pixeldataIn[cs_mr.snr_slice].astype(float))
        #data = self._movingaverage(cs_mr.pixeldataIn[cs_mr.snr_slice].astype(float),3)

        # Step 3
        N = 2

        wid = cs_mr.pixeldataIn[cs_mr.snr_slice].shape[0]
        hei = cs_mr.pixeldataIn[cs_mr.snr_slice].shape[1]
        contourimage = np.zeros((wid,hei))
        image10 = np.zeros((wid,hei)) # for rad10 calculation; grey2
        image20 = np.zeros((wid,hei)) # for rad20 calculation: grey1+grey2+grey3

        gUndef = 0
        gBlack = 1
        gGrey1 = 2
        gGrey2 = 3
        gGrey3 = 4
        gGrey4 = 5
        gGrey5 = 6
        gWhite = 7
        T = 10.*B
        S = .1*(C-B) # 10% for Head coil
        #rad_10%: !grey2
        #rad_20%: !(grey1+grey2+grey3)

        counts = np.zeros(gWhite+1,dtype=int)
        if N == 2: # Use the whole field of view for Head Coil
            for y in range(hei):
                for x in range(wid):
                    pval = data[x,y]
                    if(pval<T): # undef
                        counts[gUndef] += 1
                        continue
                    if(pval< C-N*S): # black
                        contourimage[x,y] = gBlack
                        counts[gBlack] += 1
                        continue
                    if(pval< C-(N-1)*S): # grey1
                        contourimage[x,y] = gGrey1
                        counts[gGrey1] += 1
                        image20[x,y] = 1
                        continue
                    if(pval< C+(N-1)*S): # grey2
                        contourimage[x,y] = gGrey2
                        counts[gGrey2] += 1
                        image10[x,y] = 1
                        image20[x,y] = 1
                        continue
                    if(pval< C+N*S): # grey3
                        contourimage[x,y] = gGrey3
                        counts[gGrey3] += 1
                        image20[x,y] = 1
                        continue
                    contourimage[x,y] = gWhite
                    counts[gWhite] += 1

            # calculated rad10 and rad20 by euclidan distance transform
            image10 = scind.morphology.distance_transform_edt(image10)
            rad10 = np.max(image10)
            xmid10,ymid10 = np.unravel_index(image10.argmax(),image10.shape)
            image20 = scind.morphology.distance_transform_edt(image20)
            rad20 = np.max(image20)
            xmid20,ymid20 = np.unravel_index(image20.argmax(),image20.shape)

            Ntot = np.sum(counts)-counts[gUndef]

            cs_mr.ffu_Ntot = Ntot
            cs_mr.ffu_mid10 = [xmid10,ymid10]
            cs_mr.ffu_mid20 = [xmid20,ymid20]

            # Report values
            cs_mr.ffu_TCm20    = 100.*counts[gBlack]/Ntot
            cs_mr.ffu_Cm20Cm10 = 100.*counts[gGrey1]/Ntot
            cs_mr.ffu_Cm10Cp10 = 100.*counts[gGrey2]/Ntot
            cs_mr.ffu_Cp10Cp20 = 100.*counts[gGrey3]/Ntot
            cs_mr.ffu_Cp20MAX  = 100.*counts[gWhite]/Ntot
            cs_mr.ffu_rad10 = self.pix2phantommm(cs_mr,rad10)
            cs_mr.ffu_rad20 = self.pix2phantommm(cs_mr,rad20)

        #
        """ NEMA FFU
         int_unif = (max pixelval-min pixelval)/(max pixelval+min pixelval)*100
         ROI =
         The image is filtered to reduce noise influences on the values of the maximum pixel
            value and the minimum pixel value. The specification area is a circular ROI, with the
            centre of the image as centre and with a radius of 150mm for the head coil.

         """
        # 3. Smoothing: moving average of image
        data = self._lowpassfilter(cs_mr.pixeldataIn[cs_mr.snr_slice].astype(float))
        #data = self._movingaverage(cs_mr.pixeldataIn[cs_mr.snr_slice].astype(float),3)

        # 3.3 select data inside circle only
        wid = data.shape[0]
        x,y = np.indices((wid, wid))
        mid = wid/2
        rad = 150/2 # QA3: 176/2, QA2:150/2
        #rad = 150/2 #176/2 #int(self.phantommm2pix(cs_mr,175./2.))
        mask = ((x-mid)**2 + (y-mid)**2 ) < rad**2

        maxval = np.max(data[mask])
        minval = np.min(data[mask])
        cs_mr.ffu_mid_linunif = [mid,mid]
        cs_mr.ffu_rad_linunif = rad

        # report value
        cs_mr.ffu_lin_unif = 100.*(maxval-minval)/(maxval+minval)
        cs_mr.resultimage['FFU'] = copy.deepcopy(contourimage)

        error = False
        return error


    def SpatialLinearity(self,cs_mr):
        """
        The spatial linearity section of the head phantom is a regular array of 45
        discs. The diameter and spacing between the discs are:
            diameter 5 mm, distance between discs 25 mm.

        The program evaluates an image of the spatial linearity section as follows:
        1) The 'theoretical' array of discs is matched to the array in the image.
        2) For every disc the shift of the centre of the disc in the image with respect to the
        theoretical position is determined in both the horizontal and vertical direction:
        A positive horizontal shift means a shift to the right and a positive vertical shift
        means a shift to the top of the image.
        The accuracy of the program is 1/4 pixel.
        3) For every pair of adjacent discs the program determines the so-called differential
        linearity:
            differential linearity = 100 * ( (actual dist. / theoretical dist.) - 1)
        This means that if the actual distance matches the theoretical one, the differential linearity will
        be 0%.
        To specify the spatial linearity, Philips defined parameters which are calculated from
        the shifts and differential linearity numbers:
            integral linearity avg: average value of shifts
            integral linearity std: standard deviation
            max_R: maximum shift to the right/upwards
            min_L: maximum shift to the left/downwards
            diff. linearity avg: average of differential linearity values
            diff. linearity std: standard deviation
            max: maximum value
            min: minimum value

        Both the integral and differential linearity, maximum and minimum values are calculated
        in both the horizontal and vertical direction of the image.
        Apart from the above parameters the size of the phantom in both horizontal and vertical
        direction is determined:
            size_hor: distance between the outer discs on the horizontal axis.
            size_ver: distance between the outer discs on the vertical axis.

        Note that a possible scaling in the matching of theoretical grid should be prevented; the 
          theoretical distances should remain 25 mm.

        Workflow:
            1. Make grid of theoretical positions
            2. Find real location starting from theoretical ones, acc 1/4 pixel
            3. Find rigid transformation, between theoretical positions and real locations
            4. Apply transform to theoretical positions
            5. Calculate all parameters
        """
        error = True
        if cs_mr is None or cs_mr.dcmInfile is None or cs_mr.pixeldataIn is None:
            print("[SpatialLinearity] Error: Not a valid PiQT struct")
            return error

        # image sequence
        cs_mr.lin_slice = self.ImageSliceNumber(cs_mr,cs_mr.piqttest)
        if(cs_mr.lin_slice <0):
            print("[SpatialLinearity] ERROR: Test", lit.stTestSpatialLinearity,"not available for given image")
            return error

        # coiltype
        coiltype = self.CoilType(cs_mr,cs_mr.lin_slice)
        if coiltype == lit.stUnknown or coiltype == lit.stCoilSurface:
            print("[SpatialLinearity] ERROR: coiltype not recognized")
            return error

        # 1. grid of theoretical positions
        wid = cs_mr.pixeldataIn[cs_mr.snr_slice].shape[0]
        hei = cs_mr.pixeldataIn[cs_mr.snr_slice].shape[1]
        xmid = (wid-1)/2.
        ymid = (hei-1)/2.
        ncent = 7
        defdistmm = 25.
        defdiamm  = 5.

        dxy = self.phantommm2pix(cs_mr,defdistmm)
        x0 = xmid - (ncent-1)/2*dxy
        y0 = ymid - (ncent-1)/2*dxy
        # build 2d array of px coordinates, but remove corners
        pos_gt =[[ [x*dxy+x0,y*dxy+y0] for x in range(ncent)] for y in range(ncent)]
        pos_gt[ 0][ 0] = []
        pos_gt[-1][ 0] = []
        pos_gt[-1][-1] = []
        pos_gt[ 0][-1] = []

        # 2. real location starting from theoretical ones, acc 1/4 pixel
        pos_found = copy.deepcopy(pos_gt)
        error,pos_found = mymath.FindCenters2D(pos_found,cs_mr.pixeldataIn[cs_mr.lin_slice],self.phantommm2pix(cs_mr,defdistmm/4.),self.phantommm2pix(cs_mr,defdiamm),minimod=False)

        # 3. Find rigid transformation, between theoretical positions and real locations
        ## turn grid into list for matching
        fit_posgt    = []
        fit_posfound = []
        for y in range(ncent):
            for x in range(ncent):
                gt = pos_gt[y][x]
                if(len(gt)==0):
                    continue
                found = pos_found[y][x]
                fit_posgt.append(gt)
                fit_posfound.append(found)

        # rigid transformation
        trn = mymath.RigidTransform(fit_posgt, fit_posfound)

        # 4. Apply transform to theoretical positions
        for y in range(ncent):
            for x in range(ncent):
                gt = pos_gt[y][x]
                if len(gt)==0:
                    continue
                pos_gt[y][x] = trn.apply(gt) # apply rigid transformation

        # 5. do calculations
        # find shifts between groundtruth and found
        shiftx = []
        shifty = []
        for y in range(ncent):
            for x in range(ncent):
                gt = pos_gt[y][x]
                if len(gt)==0:
                    continue
                found = pos_found[y][x]
                shiftx.append(found[0]-gt[0])
                shifty.append(found[1]-gt[1])
                #print(gt[0],gt[1],found[0],found[1],shiftx[-1],shifty[-1])

        # find distances between adjacent discs
        difflinx = []
        ## theoretical distance should be 25 mm
        for y in range(ncent):
            for x in range(ncent-1):
                pos0 = pos_found[y][x]
                pos1 = pos_found[y][x+1]
                if len(pos0) == 0 or len(pos1)==0:
                    continue
                dist = np.sqrt( (pos1[0]-pos0[0])**2+ (pos1[1]-pos0[1])**2)

                thpos0 = pos_gt[y][x]
                thpos1 = pos_gt[y][x+1]
                # theoretical distance should be 25 mm
                thdist = np.sqrt( (thpos1[0]-thpos0[0])**2+ (thpos1[1]-thpos0[1])**2)

                difflinx.append(100.*(dist/thdist -1.))
                #print(x,y,dist,thdist,difflinx[-1],self.pix2phantommm(cs_mr,thdist))

        diffliny = []
        for y in range(ncent-1):
            for x in range(ncent):
                pos0 = pos_found[y][x]
                pos1 = pos_found[y+1][x]
                if len(pos0) == 0 or len(pos1)==0:
                    continue
                dist = np.sqrt( (pos1[0]-pos0[0])**2+ (pos1[1]-pos0[1])**2)

                thpos0 = pos_gt[y][x]
                thpos1 = pos_gt[y+1][x]
                # theoretical distance should be 25 mm
                thdist = np.sqrt( (thpos1[0]-thpos0[0])**2+ (thpos1[1]-thpos0[1])**2)

                diffliny.append(100.*(dist/thdist -1.))


        # 6. NEMA
        """
        The NEMA spatial linearity is defined by the maximum percentage of differential linearity
        between the following disk pairs:
            2-44
            3-43
            4-42
            7-39
            11-35
            13-33
            19-27
            20-26
            NEMA integral linearity = max differential linearity (N_1...N_8)

        """
        dictdiscindex = {}
        idx = 1
        for y in range(ncent):
            for x in range(ncent):
                gt = pos_gt[y][x]
                if len(gt)==0:
                    continue
                dictdiscindex.update({idx: [x,y]})
                idx += 1

        nemapairs = [
            [ 2,44],
            [ 3,43],
            [ 4,42],
            [ 7,39],
            [11,35],
            [13,33],
            [19,27],
            [20,26],
        ]

        cs_mr.lin_nema = []
        cs_mr.lin_nema_label = []
        for p in nemapairs:
            cs_mr.lin_nema_label.append("%d-%d" % (p[0],p[1]))
            id0 = dictdiscindex[p[0]]
            id1 = dictdiscindex[p[1]]
            thpos0 = pos_gt[id0[1]][id0[0]]
            thpos1 = pos_gt[id1[1]][id1[0]]
            thdistmm = self.pix2phantommm(cs_mr,np.sqrt( (thpos1[0]-thpos0[0])**2+ (thpos1[1]-thpos0[1])**2))
            pos0 = pos_found[id0[1]][id0[0]]
            pos1 = pos_found[id1[1]][id1[0]]
            dist = self.pix2phantommm(cs_mr,np.sqrt( (pos1[0]-pos0[0])**2+ (pos1[1]-pos0[1])**2) )
            cs_mr.lin_nema.append(100.*(dist/thdistmm -1.))

        ## for gui
        cs_mr.lin_diampx   = self.phantommm2pix(cs_mr,defdiamm)
        cs_mr.lin_posgt    = []
        cs_mr.lin_posfound = []
        for y in range(ncent):
            for x in range(ncent):
                gt = pos_gt[y][x]
                if len(gt)==0:
                    continue
                found = pos_found[y][x]
                cs_mr.lin_posgt.append(gt)
                cs_mr.lin_posfound.append(found)

        # Report values
        cs_mr.lin_phantomshift  = [ self.pix2phantommm(cs_mr,p) for p in  trn.getShift()]
        ## make sure we report a value between -45 and +45 degrees
        rotdeg = 180.*(trn.getRotation()/np.pi)
        cs_mr.lin_phantomrotdeg = rotdeg
        ## distance between horizontal and vertical outer discs
        nc2 = int((ncent-1)/2)
        dx = pos_found[nc2][-1][0]-pos_found[nc2][0][0]
        dy = pos_found[nc2][-1][1]-pos_found[nc2][0][1]
        cs_mr.lin_sizehor = self.pix2phantommm(cs_mr,np.sqrt(dx**2+dy**2))
        dx = pos_found[-1][nc2][0]-pos_found[0][nc2][0]
        dy = pos_found[-1][nc2][1]-pos_found[0][nc2][1]
        cs_mr.lin_sizever = self.pix2phantommm(cs_mr,np.sqrt(dx**2+dy**2))
        ## avg of abs and stdev of abs and min/max of normal shifts
        cs_mr.lin_intshiftavg  = [self.pix2phantommm(cs_mr,np.mean([abs(x) for x in shiftx])),self.pix2phantommm(cs_mr,np.mean([abs(x) for x in shifty]))]
        cs_mr.lin_intshiftsdev = [self.pix2phantommm(cs_mr,np.std([abs(x) for x in shiftx])),self.pix2phantommm(cs_mr,np.std([abs(x) for x in shifty]))]
        cs_mr.lin_shiftmax     = [self.pix2phantommm(cs_mr, np.max(shiftx)), self.pix2phantommm(cs_mr,-np.min(shifty))]
        cs_mr.lin_shiftmin     = [self.pix2phantommm(cs_mr,-np.min(shiftx)), self.pix2phantommm(cs_mr, np.max(shifty))]
        ## avg and stdev and min/max of differential linearity horz and vert
        cs_mr.lin_intdiffavg =  [np.mean(difflinx),np.mean(diffliny)]
        cs_mr.lin_intdiffsdev = [np.std(difflinx), np.std(diffliny)]
        cs_mr.lin_intdiffmax =  [ np.max(difflinx),  np.max(diffliny)]
        cs_mr.lin_intdiffmin =  [ np.min(difflinx),  np.min(diffliny)]
        ## nema
        cs_mr.lin_nema_max = np.max([np.max(cs_mr.lin_nema),-np.min(cs_mr.lin_nema)])
        """
        m/p_angle	89.98	S89 - 91
        preparation angle is probably Flip Angle (0018,1314), but measurement angle?
        """
        ## only for displaying
        # smoothing to get rid of noise and give max respons over avg disc
        data = cs_mr.pixeldataIn[cs_mr.lin_slice]
        sigma = self.phantommm2pix(cs_mr,defdiamm/2.)
        cs_mr.resultimage['LIN'] = scind.gaussian_filter(data.astype(float), sigma,mode='constant')

        error = False
        return error

    def DICOMInfo(self,cs,info='dicom',imslice=0):
        # Different from ImageJ version; tags "0008","0104" and "0054","0220"
        #  appear to be part of sequences. This gives problems (cannot be found
        #  or returning whole sequence blocks)
        # Possibly this can be solved by using if(type(value) == type(dicom.sequence.Sequence()))
        #  but I don't see the relevance of these tags anymore, so set them to NO
        if info == "dicom":
            dicomfields = [
                ["0010,0010", "Patients Name"], # PIQT
                ["0018,1030", "Protocol Name"],  # QA1S:MS,SE
                ["0008,0021", "Series Date"],
                ["0008,0031", "Series Time"],# no ScanTime 0008,0032 in EnhancedDicom
                ["0018,1250", "Receive Coil Name"], # Q-Body
                ["0018,1251", "Transmit Coil Name"], # B
                ["0018,9051", "Transmit Coil Type"], # SURFACE
                ["0018,0095", "Pixel Bandwidth"], # 219
                ["0018,0020", "Scanning Sequence"], # SE
                ["0018,0021", "Scanning Variant"], # SS
                ["2005,1011", "Image_Type"], # M
                ["0018,0081", "Echo Time"], # 50
                ["0020,0012", "Acquisition Number"], # 5
                ["0018,0086", "Echo Number(s)"], # 1
                ["2001,1081", "Dyn_Scan_No"], # ?1
                ["0020,0013", "Instance Number"], # 1 slice no?
                ["2001,105f,2005,1079", "Dist_sel"], # -16.32
                ["2001,1083", "Central_freq"], # 63.895241 (MHz)
                ["0018,1020", "SoftwareVersions"], 
            ]
        elif info == "id":
            dicomfields = [
                ["0018,1030", "Protocol Name"],  # QA1S:MS,SE
                ["0008,0021", "Series Date"],
                ["0008,0031", "Series Time"],
            ]
        results = []
        for df in dicomfields:
            key = df[0]
            value = self.readDICOMtag(cs,key,imslice)
            if key=="0018,1020" and len(value)>1:
                value = '_'.join(value)
            results.append( (df[1],value) )

        return results

    def DetermineScanID(self,cs):
        # 2. determine QA1/QA2/QA3
        dicomkey = ["0018,1030", "ProtocolName"]  # QA1S:MS,SE
        value = self.readDICOMtag(cs,dicomkey[0],0)
        if 'QA1' in value:
            cs.scanID = 'QA1'
        elif 'QA2' in value:
            cs.scanID = 'QA2'
        elif 'QA3' in value:
            cs.scanID = 'QA3'
        else:
            cs.scanID = lit.stUnknown

        return cs.scanID == lit.stUnknown

    def saveResultImage(self,cs_mr,kind,fname):
        # construct an result image with ROIs is applicable

        # make a palette, mapping intensities to greyscale
        pal = np.arange(0,256,1,dtype=np.uint8)[:,np.newaxis] * \
            np.ones((3,),dtype=np.uint8)[np.newaxis,:]
        # but reserve the first for red/green/blue for markings
        pal[0] = [255,0,0]
        pal[1] = [0,255,0]
        pal[2] = [0,0,255]


        im = None
        rois = [] # list of rois (type,color,roi)
        if kind == 'FFU': # flood field uniformity
            # use countour image as base
            # convert to 8-bit palette mapped image with lowest palette value used = 1
            im = toimage(cs_mr.resultimage['FFU'].transpose(),low=3,pal=pal) # MODULE EXPECTS PYQTGRAPH DATA: X AND Y ARE TRANSPOSED!

            # add ROIs for SNR measurement
            for r in cs_mr.snr_rois:
                rois.append( ('rectangle',0,r))

            # add roi for artefact measurements
            rois.append( ('circle',0,cs_mr.artefact_roi))

            # add ROIs indication uniformity measures
            rois.append( ('circle',2,[cs_mr.ffu_mid10[0],cs_mr.ffu_mid10[1],self.phantommm2pix(cs_mr,cs_mr.ffu_rad10)]))
            rois.append( ('circle',2,[cs_mr.ffu_mid20[0],cs_mr.ffu_mid20[1],self.phantommm2pix(cs_mr,cs_mr.ffu_rad20)]))
            rois.append( ('circle',2,[cs_mr.ffu_mid_linunif[0],cs_mr.ffu_mid_linunif[1],self.phantommm2pix(cs_mr,cs_mr.ffu_rad_linunif)]))
        elif kind == 'SLP': # slice profile
            # use special image if available, else generate one
            if 'SLP' in cs_mr.resultimage:
                # convert to 8-bit palette mapped image with lowest palette value used = 1
                im = toimage(cs_mr.resultimage['SLP'].transpose(),low=3,pal=pal) # MODULE EXPECTS PYQTGRAPH DATA: X AND Y ARE TRANSPOSED!
            else:
                im = toimage(cs_mr.pixeldataIn[cs_mr.sp_slice].transpose(),low=3,pal=pal) # MODULE EXPECTS PYQTGRAPH DATA: X AND Y ARE TRANSPOSED!

            # add locations of detected pins
            for pos in cs_mr.sp_pins:
                rois.append( ('circle',1,[pos[0],pos[1],self.phantommm2pix(cs_mr,cs_mr.sp_diamm/2.)]) )

            # add boxes for measurements
            for r in cs_mr.sp_rois:
                rois.append( ('rectangle',0,r))

        elif kind == 'LIN': # spatial linearity
            if 'SLP' in cs_mr.resultimage:
                # convert to 8-bit palette mapped image with lowest palette value used = 1
                im = toimage(cs_mr.resultimage['LIN'].transpose(),low=3,pal=pal) # MODULE EXPECTS PYQTGRAPH DATA: X AND Y ARE TRANSPOSED!
            else:
                im = toimage(cs_mr.pixeldataIn[cs_mr.lin_slice].transpose(),low=3,pal=pal) # MODULE EXPECTS PYQTGRAPH DATA: X AND Y ARE TRANSPOSED!

            # add locations of GT pins
            for pos in cs_mr.lin_posgt:
                rois.append( ('circle',0,[pos[0],pos[1],cs_mr.lin_diampx/2.]) )

            # add locations of detected pins
            for pos in cs_mr.lin_posfound:
                rois.append( ('circle',2,[pos[0],pos[1],cs_mr.lin_diampx/2.]) )

        elif kind == 'MTF':
            if 'MTF' in cs_mr.resultimage:
                # convert to 8-bit palette mapped image with lowest palette value used = 1
                im = toimage(cs_mr.resultimage['MTF'].transpose(),low=3,pal=pal) # MODULE EXPECTS PYQTGRAPH DATA: X AND Y ARE TRANSPOSED!
            else:
                im = toimage(cs_mr.pixeldataIn[cs_mr.mtf_slice].transpose(),low=3,pal=pal) # MODULE EXPECTS PYQTGRAPH DATA: X AND Y ARE TRANSPOSED!

            # add boxes for measurements
            for ix,r in enumerate(cs_mr.mtf_rois):
                if ix<2: # snr rois
                    color = 0 
                elif ix<6: # quadrants to find angle of phantom
                    color = 1
                else: # mtf rois
                    color = 2
                rois.append( ('rectangle',color,r))

        if im is None:
            return im

        # now draw all rois in reserved color
        if len(rois)>0:
            draw = ImageDraw.Draw(im)
            for ty,color,r in rois:
                snr_rois   = [] # xy roi definitions # format: x0,wid, yo,hei
                if ty == 'rectangle':
                    x0 = int(r[0]+.5)
                    x1 = int(r[0]+r[1]+.5)
                    y0 = int(r[2]+.5)
                    y1 = int(r[2]+r[3]+.5)
                    draw.rectangle([(x0,y0),(x1,y1)],outline=color)
                elif ty == 'polygon':
                    roi =[]
                    for x,y in r:
                        roi.append( (int(x+.5),int(y+.5)))
                    draw.polygon(roi,outline=color)
                elif ty == 'circle':
                    x0  = int (r[0]+.5)
                    y0  = int (r[1]+.5)
                    rad = int (r[2]+.5)
                    draw.ellipse(( x0-rad,y0-rad,x0+rad,y0+rad), outline=color)
            del draw

        # convert to RGB for JPG, cause JPG doesn't do PALETTE and PNG is much larger
        im = im.convert("RGB")

        imsi = im.size
        if max(imsi)>2048:
            ratio = 2048./max(imsi)
            im = im.resize( (int(imsi[0]*ratio+.5), int(imsi[1]*ratio+.5)),Image.ANTIALIAS)
        im.save(fname)
