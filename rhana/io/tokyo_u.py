import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import re
from typing import IO as IO_Type, Optional, Union, Dict, List
import yaml
import io

with (Path(__file__).parent/"config/meta_parameters_map.yml").open() as f:
    META_PARAMS_MAP = yaml.load(f, yaml.FullLoader)

META_PARAMS_MAP_REV = {}
for main_k, main_v in META_PARAMS_MAP.items():
    for sub_k, sub_v in main_v.items():
        META_PARAMS_MAP_REV[sub_v['name']] = f"{main_k}-{sub_k} ({sub_v['unit']})"

def read_binary(buffer, decode_format, num_bundle=1, return_first=True):
    """
    see https://docs.python.org/3/library/struct.html#format-characters for possible decode_format
    """


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

def summarize_files(rheed_files):
    rheed_files = [ Path(file) for file in rheed_files ]
    rheed_files = [ file for file in rheed_files if file.suffix == ".dir" ]
    summary = defaultdict(lambda : [])
    for dir_file in rheed_files:
        bin_file = dir_file.parent / (dir_file.stem + 'bin')
            
        library_name = dir_file.stem
        res = re.findall("(\w+\d+)-RHEED-(\d+)", library_name)
        if res:
            project_name, num = res[0]
            summary[project_name].append(num)
    
    for k in summary.keys():
        summary[k] = sorted(summary[k])

    return summary 


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
    gunh : Optional[int] = None
    tilth : Optional[int] = None
    gunv : Optional[int] = None
    tiltv : Optional[int] = None
    offgunh : Optional[int] = None
    offtilth : Optional[int] = None
    offgunv : Optional[int] = None
    offtiltv : Optional[int] = None
    focus : Optional[int] = None
    meta : Optional[Dict] = field(default_factory=dict)

    @classmethod
    def from_streamreader_V5(cls, dir_fo:IO_Type, streamreader:"RHEEDStreamReader", frame_index:int):
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

        gunh = read_binary(read_fo, "i", return_first = True)
        tilth = read_binary(read_fo, "i", return_first = True)
        gunv = read_binary(read_fo, "i", return_first = True)
        tiltv = read_binary(read_fo, "i", return_first = True)
        offgunh = read_binary(read_fo, "i", return_first = True)
        offtilth = read_binary(read_fo, "i", return_first = True)
        offgunv = read_binary(read_fo, "i", return_first = True)
        offtiltv = read_binary(read_fo, "i", return_first = True)
        focus = read_binary(read_fo, "i", return_first = True)

        
        header = cls(
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

            gunh = gunh,
            tilth = tilth,
            gunv = gunv,
            tiltv = tiltv,
            offgunh = offgunh,
            offtilth = offtilth,
            offgunv = offgunv,
            offtiltv = offtiltv,
            focus = focus
        )
        
        # dynamically created chamber variables
        for param_name in streamreader.param_names:
            param_value = read_binary(read_fo, "f", return_first = True)
            # setattr(header, param_name, param_value)
            if param_name in META_PARAMS_MAP_REV:
                param_name = META_PARAMS_MAP_REV[param_name]
            header.meta[param_name] = param_value

        return header

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

class BaseRHEEDStreamReader:
    _params_name_length = 4
    _main_file_header_size = 16
    _frame_storage_fmt = "H"
        
    def __init__(self, directory_fo, stream_fo, directory_size, is_close=False):
        
        # self.directory_filename = Path(directory_filename)
        # self.stream_filename = Path(stream_filename)
        # self.name = self.directory_filename.stem
        
        # self.directory_size = self.directory_filename.stat().st_size
        # self.open()
        # self.read_header()

        self.directory_fo = directory_fo
        self.stream_fo = stream_fo
        self.directory_size = directory_size
        self.is_close = is_close

        self.read_header()
    
    def _raise_version_error(self):
        raise ValueError(f"Version {self.version} is not supported")        
        
    def close(self,):
        self.directory_fo.close()
        self.stream_fo.close()
        self.is_close = True
                        
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
        
        def _read_fblock_v5():                
            self.dblock_sz = read_binary(read_fo, "I")
            self.maxframes = (self.directory_size - 16 - self.dblock_sz) // self.frame_block_sz

            _ = read_binary(read_fo, "I")
            npars = read_binary(read_fo, "I")

            param_names = read_binary(read_fo, f"{self._params_name_length * npars:d}s",  return_first=True).decode()
            self.param_names = [ param_names[i*self._params_name_length:i*self._params_name_length+self._params_name_length] for i in range(npars) ]
            self.param_def = read_block(read_fo, "I", "s", lambda x:x.decode())
        
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

        if self.version == 5:
            # v5 and v4 is same
            _read_fblock_v5()                        
        elif self.version == 4:
            _read_fblock_v4()
        elif self.version == 3:
            _read_fblock_v3()
        elif self.version == 2:
            _read_fblock_v2()            
        elif self.version == 1:
            _read_fblock_v1()
        else:
            self._raise_version_error()
            # print(f"Version {self.version} is not supported.")
            
        return self
    
    
    def read_frame_header(self, frame_index):
        if self.version == 5:
            return RHEEDFrameHeader.from_streamreader_V5(self.directory_fo, self, frame_index)
        elif self.version == 4:
            return RHEEDFrameHeader.from_streamreader_V4(self.directory_fo, self, frame_index)
        elif self.version == 3:
            return RHEEDFrameHeader.from_streamreader_V3(self.directory_fo, self, frame_index)
        elif self.version == 2:
            return RHEEDFrameHeader.from_streamreader_V2(self.directory_fo, self, frame_index)
        elif self.version == 1:
            return RHEEDFrameHeader.from_streamreader_V1(self.directory_fo, self, frame_index)
        else:
            self._raise_version_error()
            # raise ValueError(f"Version {self.version} is not supported")
        
    def read_frame_content(self, frame_index, frame_header):
        read_fo = self.stream_fo
        read_fo.seek(frame_header.frameoff)
        image_bin = read_fo.read(struct.calcsize(self._frame_storage_fmt)*frame_header.width*frame_header.height)
        image = np.frombuffer(image_bin, dtype=self._frame_storage_fmt, offset=0).reshape(frame_header.height, frame_header.width)
        return image
        
    
    def read_frame(self, frame_index, callback=None):
        frame_header = self.read_frame_header(frame_index)
        frame = self.read_frame_content(frame_index, frame_header)
        return frame, frame_header


    def get_beams(self):
        beams = {}
        for i in range(0, self.maxframes):
            frame_header = self.read_frame_header(i)
            # if frame_header is None: continue

            if frame_header.bpos not in beams:
                beams[frame_header.bpos] = {
                    "exposure" : frame_header.exposure,
                    "frames" : [i],
                    "gain" : frame_header.gain,
                    "width" : frame_header.width,
                    "height" : frame_header.height
                }
            else:
                beams[frame_header.bpos]["frames"].append(i)
        return beams

class RHEEDStreamReader(BaseRHEEDStreamReader):
        
    def __init__(self, directory_filename, stream_filename):
        
        self.directory_filename = Path(directory_filename)
        self.stream_filename = Path(stream_filename)
        self.name = self.directory_filename.stem        
        directory_size = self.directory_filename.stat().st_size
        directory_fo, stream_fo, is_close = self.open(self.directory_filename, self.stream_filename)
        
        super().__init__(
            directory_fo=directory_fo,
            stream_fo=stream_fo,
            directory_size=directory_size,
            is_close=is_close
        )

    @classmethod
    def from_library(cls, name, number, folder):
        folder = Path(folder)
        dir_path = folder / f"{name}-RHEED-{number}.dir"
        bin_path = folder / f"{name}-RHEED-{number}.bin"
        assert dir_path.exists(), f"{dir_path} do not exists"
        assert bin_path.exists(), f"{bin_path} do not exists"
        return cls(dir_path, bin_path)

    @staticmethod
    def summarize_folder(folder):
        # summary = defaultdict(lambda: [])
        folder = Path(folder)
        rheed_files = folder.glob("*.dir")
        return summarize_files(rheed_files)
        # for dir_file in rheed_files:
        #     bin_file = dir_file.parent / (dir_file.stem + ".bin")
        #     if bin_file.exists():
        #         library_name = dir_file.stem
        #         res = re.findall("(\w+\d+)-RHEED-(\d+)", library_name)
        #         if res:
        #             project_name, num = res[0]
        #             summary[project_name].append(num)
        
        # for k in summary.keys():
        #     summary[k] = sorted(summary[k])

        # return summary 

    def _raise_version_error(self):
        raise ValueError(f"Version {self.version} is not supported")        

    def reconnect(self):
        directory_fo, stream_fo, is_close = self.open(self.directory_filename, self.stream_filename)
        self.directory_fo = directory_fo
        self.stream_fo = stream_fo
        self.is_close = is_close

    def open(self, directory_filename, stream_filename):
        directory_fo = open(directory_filename, "rb")
        stream_fo = open(stream_filename, "rb")
        is_close = False

        return directory_fo, stream_fo, is_close
                                
class SFTPRHEEDStreamReader(BaseRHEEDStreamReader):
    def __init__(self, directory_filename, stream_filename, sftp_client):
        
        self.directory_filename = Path(directory_filename)
        self.stream_filename = Path(stream_filename)
        self.name = self.directory_filename.stem

        self.sftp_client = sftp_client
        directory_size = sftp_client.lstat(stream_filename).st_size

        directory_fo, stream_fo, is_close = self.open(self.directory_filename, self.stream_filename)
        
        super().__init__(
            directory_fo=directory_fo,
            stream_fo=stream_fo,
            directory_size=directory_size,
            is_close=is_close
        )

    @classmethod
    def from_library(cls, name, number, folder, sftp_client):
        folder = Path(folder)
        dir_path = folder / f"{name}-RHEED-{number}.dir"
        bin_path = folder / f"{name}-RHEED-{number}.bin"
        assert dir_path.exists(), f"{dir_path} do not exists"
        assert bin_path.exists(), f"{bin_path} do not exists"
        return cls(dir_path, bin_path, sftp_client)

    @staticmethod
    def summarize_folder(folder):
        # summary = defaultdict(lambda: [])
        folder = Path(folder)
        rheed_files = folder.glob("*.dir")
        return summarize_files(rheed_files)
        # for dir_file in rheed_files:
        #     bin_file = dir_file.parent / (dir_file.stem + ".bin")
        #     if bin_file.exists():
        #         library_name = dir_file.stem
        #         res = re.findall("(\w+\d+)-RHEED-(\d+)", library_name)
        #         if res:
        #             project_name, num = res[0]
        #             summary[project_name].append(num)
        
        # for k in summary.keys():
        #     summary[k] = sorted(su
        # .mmary[k])

        # return summary 

    def _raise_version_error(self):
        raise ValueError(f"Version {self.version} is not supported")        

    def reconnect(self):
        directory_fo, stream_fo, is_close = self.open(self.directory_filename, self.stream_filename)
        self.directory_fo = directory_fo
        self.stream_fo = stream_fo
        self.is_close = is_close

    def open(self, directory_filename, stream_filename):
        directory_fo = self.sftp_client.open(str(directory_filename), mode="rb")
        stream_fo = self.sftp_client.open(str(stream_filename), mode="rb")
        is_close = False

        return directory_fo, stream_fo, is_close
