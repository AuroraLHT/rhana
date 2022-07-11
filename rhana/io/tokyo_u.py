import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass


def read_binary(buffer, decode_format, num_bundle=1, return_first=True):
    size = struct.calcsize(decode_format) * num_bundle
    binary = buffer.read(size)
    if num_bundle == 1:
        vars = struct.unpack(decode_format, binary)
        if return_first:
            vars = vars[0]
        return vars
    else:
        bundles = [ vars for i, vars in enumerate(struct.iter_unpack(decode_format, binary)) ]
        if return_first:
            return [ e[0] for e in bundles ]
        else:
            return bundles
        

def read_block(buffer, blocksz_fmt, content_fmt, callback=None):
    blksz = read_binary(buffer, decode_format=blocksz_fmt, num_bundle=1, return_first = True)
    content = read_binary(buffer, f"{blksz:d}{content_fmt}", num_bundle=1, return_first = True)
    
    if callable(callback):
        return callback(content)
    else:
        content        
        
@dataclass
class RHEEDFrameHeader:
    frameoff : int
    tstamp : int
    rstamp : int
    frameindex : int
    rframeindex : int
    ccdwidth : int
    ccdheight : int
    width : int
    height : int
    offx : int
    offy : int
    nbits : int
    binning : int
    decimation : int
    exposure : int
    gain : int
    bpos : int
    bframe : int
    
    @classmethod
    def from_streamreader_V4(cls, dir_fo, streamreader, frame_index):
        read_fo = dir_fo

        read_fo.seek(16 + streamreader.dblock_sz + frame_index * streamreader.frame_block_sz, 0)    

        frameoff = read_binary(read_fo, "L", return_first = True)
        tstamp = read_binary(read_fo, "L", return_first = True)
        rstamp = read_binary(read_fo, "L", return_first = True)
        read_binary(read_fo, "I", return_first = True)
        frameindex = read_binary(read_fo, "I", return_first = True)
        rframeindex = read_binary(read_fo, "I", return_first = True)
        ccdwidth = read_binary(read_fo, "I", return_first = True)
        ccdheight = read_binary(read_fo, "I", return_first = True)
        width = read_binary(read_fo, "I", return_first = True)
        height = read_binary(read_fo, "I", return_first = True)
        offx = read_binary(read_fo, "I", return_first = True)
        offy = read_binary(read_fo, "I", return_first = True)
        nbits = read_binary(read_fo, "I", return_first = True)
        binning = read_binary(read_fo, "I", return_first = True)
        decimation = read_binary(read_fo, "I", return_first = True)
        exposure = read_binary(read_fo, "I", return_first = True)
        gain = read_binary(read_fo, "I", return_first = True)
        bpos = read_binary(read_fo, "I", return_first = True)
        bframe = read_binary(read_fo, "I", return_first = True)

        
        header =  cls(
            frameoff = frameoff,
            tstamp = tstamp,
            rstamp = rstamp,
            frameindex = frameindex,
            rframeindex = rframeindex,
            ccdwidth = ccdwidth,
            ccdheight = ccdheight,
            width = width,
            height = height,
            offx = offx,
            offy = offy,
            nbits = nbits,
            binning = binning,
            decimation = decimation,
            exposure = exposure,
            gain = gain,
            bpos = bpos,
            bframe = bframe,
        )
        
        # dynamically created chamber variables
        for param_name in streamreader.param_names:
            param_value = read_binary(read_fo, "f", return_first = True)
            setattr(header, param_name, param_value)
        
        return header
    
    @classmethod
    def from_streamreader_V3(cls, dir_fo, streamreader, frame_index):
        read_fo = dir_fo

        read_fo.seek(16 + frame_index * streamreader.fblock_sz, 0)    

        frameoff = read_binary(read_fo, "L", return_first = True)
        tstamp = read_binary(read_fo, "L", return_first = True)
        rstamp = read_binary(read_fo, "L", return_first = True)
        read_binary(read_fo, "I", return_first = True)
        frameindex = read_binary(read_fo, "I", return_first = True)
        rframeindex = read_binary(read_fo, "I", return_first = True)
        ccdwidth = read_binary(read_fo, "I", return_first = True)
        ccdheight = read_binary(read_fo, "I", return_first = True)
        width = read_binary(read_fo, "I", return_first = True)
        height = read_binary(read_fo, "I", return_first = True)
        offx = read_binary(read_fo, "I", return_first = True)
        offy = read_binary(read_fo, "I", return_first = True)
        nbits = read_binary(read_fo, "I", return_first = True)
        binning = read_binary(read_fo, "I", return_first = True)
        decimation = read_binary(read_fo, "I", return_first = True)
        exposure = read_binary(read_fo, "I", return_first = True)
        gain = read_binary(read_fo, "I", return_first = True)
        bpos = read_binary(read_fo, "I", return_first = True)
        bframe = read_binary(read_fo, "I", return_first = True)
        
        header =  cls(
            frameoff = frameoff,
            tstamp = tstamp,
            rstamp = rstamp,
            frameindex = frameindex,
            rframeindex = rframeindex,
            ccdwidth = ccdwidth,
            ccdheight = ccdheight,
            width = width,
            height = height,
            offx = offx,
            offy = offy,
            nbits = nbits,
            binning = binning,
            decimation = decimation,
            exposure = exposure,
            gain = gain,
            bpos = bpos,
            bframe = bframe,
        )
                
        return header
    
    @classmethod
    def from_streamreader_V2(cls, dir_fo, streamreader, frame_index):
        read_fo = dir_fo

        read_fo.seek(16 + frame_index * streamreader.fblock_sz, 0)    

        frameoff = read_binary(read_fo, "L", return_first = True)
        tstamp = read_binary(read_fo, "L", return_first = True)
        rstamp = read_binary(read_fo, "L", return_first = True)
        read_binary(read_fo, "I", return_first = True)
        frameindex = read_binary(read_fo, "I", return_first = True)
        rframeindex = read_binary(read_fo, "I", return_first = True)
        width = read_binary(read_fo, "I", return_first = True)
        height = read_binary(read_fo, "I", return_first = True)        
        ccdwidth = width
        ccdheight = height
        offx = read_binary(read_fo, "I", return_first = True)
        offy = read_binary(read_fo, "I", return_first = True)
        nbits = read_binary(read_fo, "I", return_first = True)
        binning = read_binary(read_fo, "I", return_first = True)
        decimation = read_binary(read_fo, "I", return_first = True)
        exposure = read_binary(read_fo, "I", return_first = True)
        gain = read_binary(read_fo, "I", return_first = True)
        bpos = read_binary(read_fo, "I", return_first = True)
        bframe = read_binary(read_fo, "I", return_first = True)
        
        header =  cls(
            frameoff = frameoff,
            tstamp = tstamp,
            rstamp = rstamp,
            frameindex = frameindex,
            rframeindex = rframeindex,
            ccdwidth = ccdwidth,
            ccdheight = ccdheight,
            width = width,
            height = height,
            offx = offx,
            offy = offy,
            nbits = nbits,
            binning = binning,
            decimation = decimation,
            exposure = exposure,
            gain = gain,
            bpos = bpos,
            bframe = bframe,
        )
                
        return header
    
    @classmethod
    def from_streamreader_V1(cls, dir_fo, streamreader, frame_index):
        read_fo = dir_fo

        read_fo.seek(16 + frame_index * streamreader.fblock_sz, 0)    

        frameoff = read_binary(read_fo, "L", return_first = True)
        tstamp = read_binary(read_fo, "L", return_first = True)
        rstamp = tstamp
        read_binary(read_fo, "I", return_first = True)
        frameindex = read_binary(read_fo, "I", return_first = True)
        rframeindex = read_binary(read_fo, "I", return_first = True)
        width = read_binary(read_fo, "I", return_first = True)
        height = read_binary(read_fo, "I", return_first = True)        
        ccdwidth = width
        ccdheight = height
        offx = read_binary(read_fo, "I", return_first = True)
        offy = read_binary(read_fo, "I", return_first = True)
        nbits = read_binary(read_fo, "I", return_first = True)
        binning = read_binary(read_fo, "I", return_first = True)
        decimation = read_binary(read_fo, "I", return_first = True)
        exposure = read_binary(read_fo, "I", return_first = True)
        gain = read_binary(read_fo, "I", return_first = True)
        bpos = read_binary(read_fo, "I", return_first = True)
        bframe = read_binary(read_fo, "I", return_first = True)
        
        header =  cls(
            frameoff = frameoff,
            tstamp = tstamp,
            rstamp = rstamp,
            frameindex = frameindex,
            rframeindex = rframeindex,
            ccdwidth = ccdwidth,
            ccdheight = ccdheight,
            width = width,
            height = height,
            offx = offx,
            offy = offy,
            nbits = nbits,
            binning = binning,
            decimation = decimation,
            exposure = exposure,
            gain = gain,
            bpos = bpos,
            bframe = bframe,
        )
                
        return header    
    
class RHEEDStreamReader():
    _params_name_length = 4
    _main_file_header_size = 16
    _frame_storage_fmt = "H"
        
    def __init__(self, directory_filename, rheed_stream_filename):
        
        self.directory_filename = Path(directory_filename)
        self.rheed_stream_filename = Path(rheed_stream_filename)
        self.name = directory_filename.stem
        
        self.directory_size = self.directory_filename.stat().st_size
        self.open()
        self.read_header()
            
    def open(self):
        self.directory_fo = open(self.directory_filename, "rb")
        self.rheed_stream_fo = open(self.rheed_stream_filename, "rb")
        self.close = False
        
    def close():
        self.directory_fo.close()
        self.rheed_stream_fo.close()
        self.close = True
                        
    def read_header(self):
        read_fo = self.directory_fo
        read_fo.seek(0, 0) # go back to the beginning
        
        self.ftype = read_binary(
            buffer=read_fo, 
            decode_format="4s"
        ).decode()
        
        self.version = read_binary(
            buffer=read_fo, 
            decode_format="I"
        ) // 256
        
        self.frame_block_sz = read_binary(
            buffer=read_fo, 
            decode_format="I"
        )
        
        
        
        def _read_fblock_v4():        
            self.dblock_sz = read_binary(read_fo, "I")
            self.maxframes = (self.directory_size - 16 - self.dblock_sz) // self.frame_block_sz

            _ = read_binary(read_fo, "I")
            npars = read_binary(read_fo, "I")
            
            param_names = read_binary(read_fo, f"{self._params_name_length * npars:d}s",  return_first=True).decode()
            self.param_names = [ param_names[i*self._params_name_length:i*self._params_name_length+self._params_name_length] for i in range(npars) ]

            # blksz = read_binary(read_fo, decode_format="I", num_bundle=1)
            # self.param_def = read_binary(read_fo, f"{blksz:d}s", return_first = True).decode()
            self.param_def = read_block(read_fo, "I", "s", lambda x:x.decode())

        def _read_fblock_v3():
            self.maxframes = (self.directory_size - 16) // self.frame_block_sz            

        def _read_fblock_v2():
            self.maxframes = (self.directory_size - 16) // self.frame_block_sz

        def _read_fblock_v1():
            self.maxframes = (self.directory_size - 16) // self.frame_block_sz
                        
        if self.version == 4:
            _read_fblock_v4()
        elif self.version == 3:
            _read_fblock_v3()
        elif self.version == 2:
            _read_fblock_v2()            
        elif self.version == 1:
            _read_fblock_v1()
        else:
            print(f"Version {self.version} is not supported.")
            
        return self
    
    
    def read_frame_header(self, frame_index):
        if self.version == 4:
            return RHEEDFrameHeader.from_streamreader_V4(self.directory_fo, self, frame_index)
        elif self.version == 3:
            return RHEEDFrameHeaderV3.from_streamreader_V3(self.directory_fo, self, frame_index)
        elif self.version == 2:
            return RHEEDFrameHeaderV2.from_streamreader_V2(self.directory_fo, self, frame_index)
        elif self.version == 1:
            return RHEEDFrameHeaderV1.from_streamreader_V1(self.directory_fo, self, frame_index)

        
    def read_frame_content(self, frame_index, frame_header):
        read_fo = self.rheed_stream_fo
        read_fo.seek(frame_header.frameoff)
        image_bin = read_fo.read(struct.calcsize(self._frame_storage_fmt)*frame_header.width*frame_header.height)
        image = np.frombuffer(image_bin, dtype=self._frame_storage_fmt, offset=0).reshape(frame_header.height, frame_header.width)
        return image
        
    
    def read_frame(self, frame_index, callback=None):
        frame_header = self.read_frame_header(frame_index)
        frame = self.read_frame_content(frame_index, frame_header)
        return frame, frame_header