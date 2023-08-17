function r = loadframe_updated(fname="", frameindex=0, binning=1)
  r = struct();
  r.name = fname;
  dirname = [fname ".dir"];
  fdir = dir(dirname);
  fd = fopen(dirname);
  ftype = char(fread(fd, 4, "char")');
  
  if (!strcmp(ftype, "RHED"))
    disp("Not a RHEED file");
    return;
  endif
  fver = fread(fd, 1, "uint32");
  fblocksz = fread(fd, 1, "uint32");

  r.version = fver/256;

  elseif (fver == (4*256+0))
    dblocksz = fread(fd, 1, "uint32");
    maxframes = (fdir.bytes - 16 - dblocksz) / fblocksz;
    if (frameindex >= maxframes)
      disp("Frame index out of range")
      return;
    endif

    blksz = fread(fd, 1, "uint32");
    npars = fread(fd, 1, "uint32");
    parnames = char(fread(fd, 4 * npars, "uchar")');

    blksz = fread(fd, 1, "uint32");
    r.def = char(fread(fd, blksz, "uchar")');

    fseek(fd, 16 + dblocksz + frameindex * fblocksz, SEEK_SET);
    frameoff = fread(fd, 1, "uint64");
    r.tstamp = fread(fd, 1, "uint64");
    r.rstamp = fread(fd, 1, "uint64");
    fread(fd, 1, "uint32");
    r.frameindex = fread(fd, 1, "uint32");
    r.rframeindex = fread(fd, 1, "uint32");
    r.ccdwidth = fread(fd, 1, "uint32");
    r.ccdheight = fread(fd, 1, "uint32");
    r.width = fread(fd, 1, "uint32");
    r.height = fread(fd, 1, "uint32");
    r.offx = fread(fd, 1, "uint32");
    r.offy = fread(fd, 1, "uint32");
    r.nbits = fread(fd, 1, "uint32");
    r.binning = fread(fd, 1, "uint32");
    r.decimation = fread(fd, 1, "uint32");
    r.exposure = fread(fd, 1, "uint32");
    r.gain = fread(fd, 1, "uint32");
    r.bpos = fread(fd, 1, "uint32");
    r.bframe = fread(fd, 1, "uint32");

    for i = 1:npars
      r.(parnames(i*4-3:i*4)) = fread(fd, 1, "float32");
    endfor
  elseif (fver == (5*256+0))
    dblocksz = fread(fd, 1, "uint32");
    maxframes = (fdir.bytes - 16 - dblocksz) / fblocksz;
    if (frameindex >= maxframes)
      disp("Frame index out of range")
      return;
    endif

    blksz = fread(fd, 1, "uint32");
    npars = fread(fd, 1, "uint32");
    parnames = char(fread(fd, 4 * npars, "uchar")');

    blksz = fread(fd, 1, "uint32");
    r.def = char(fread(fd, blksz, "uchar")');

    fseek(fd, 16 + dblocksz + frameindex * fblocksz, SEEK_SET);
    frameoff = fread(fd, 1, "uint64");
    r.tstamp = fread(fd, 1, "uint64");
    r.rstamp = fread(fd, 1, "uint64");
    fread(fd, 1, "uint32");
    r.frameindex = fread(fd, 1, "uint32");
    r.rframeindex = fread(fd, 1, "uint32");
    r.ccdwidth = fread(fd, 1, "uint32");
    r.ccdheight = fread(fd, 1, "uint32");
    r.width = fread(fd, 1, "uint32");
    r.height = fread(fd, 1, "uint32");
    r.offx = fread(fd, 1, "uint32");
    r.offy = fread(fd, 1, "uint32");
    r.nbits = fread(fd, 1, "uint32");
    r.binning = fread(fd, 1, "uint32");
    r.decimation = fread(fd, 1, "uint32");
    r.exposure = fread(fd, 1, "uint32");
    r.gain = fread(fd, 1, "uint32");
    r.bpos = fread(fd, 1, "uint32");
    r.bframe = fread(fd, 1, "uint32");

    r.beam = struct();
    r.gunh = fread(fd, 1, "int32");
    r.tilth = fread(fd, 1, "int32");
    r.gunv = fread(fd, 1, "int32");
    r.tiltv = fread(fd, 1, "int32");
    r.offgunh = fread(fd, 1, "int32");
    r.offtilth = fread(fd, 1, "int32");
    r.offgunv = fread(fd, 1, "int32");
    r.offtiltv = fread(fd, 1, "int32");
    r.focus = fread(fd, 1, "int32");

    for i = 1:npars
      r.(parnames(i*4-3:i*4)) = fread(fd, 1, "float32");
    endfor
  else
    disp("Unsupported file version");
    return;
  endif


  fclose(fd);

  binname = [fname ".bin"];
  fd = fopen(binname);
  fseek(fd, frameoff, SEEK_SET);
  r.image = fread(fd, [r.width r.height], "uint16")';
  fclose(fd);

  if (binning > 1)
    r.binned = struct();
    r.binned.width = floor(r.width / binning);
    r.binned.height = floor(r.height / binning);
    r.binned.image = zeros(r.binned.height, r.binned.width);
    for i= 1 : binning
      for j = 1 : binning
        r.binned.image += r.image(j:binning:r.height,i:binning:r.width);
      endfor
    endfor
  endif
endfunction
