import numpy as np

def suggest_restoration(rd, rmin=3):
    rd.get_blobs()
    db, db_i = rd.get_direct_beam()
    # ss, ss_i = rd.get_specular_spot()
    # laue_xy, laue_r = rd.get_0laue()
    laue_xy, laue_r = rd.laue_circle_analyse()
    ss = rd.ss
    laue_vector = np.array(db) - np.array(ss)

    angle =  np.rad2deg(
        np.arctan(laue_vector[1] / laue_vector[0])
    )

    return laue_xy, -angle