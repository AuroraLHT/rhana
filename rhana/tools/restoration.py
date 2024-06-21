import numpy as np

def suggest_restoration(rd, rdinst=None, rmin=3):
    if rdinst is None:
        rd.get_blobs()
        db, db_i = rd.get_direct_beam(rmin=rmin)
        laue_xy, laue_r = rd.laue_circle_analyse()
        ss = rd.ss

    else:
        db, db_i, _ = rdinst.get_direct_beam(method="top", tracker=None, track=None, direct_beam_label=3)
        laue_xy, laue_r = rdinst.laue_circle_analyse()
        ss = rdinst.ss

    # ss, ss_i = rd.get_specular_spot()
    # laue_xy, laue_r = rd.get_0laue()
    laue_vector = np.array(db) - np.array(ss)

    angle =  np.rad2deg(
        np.arctan(laue_vector[1] / laue_vector[0])
    )

    return laue_xy, -angle