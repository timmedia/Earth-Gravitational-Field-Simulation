'''geo handles geomagnetic data sets
db2 is a function that delivers a list of records. A record is a list
[altitude, latitude, longitude, B, gmlat, gmlong, sim, item], where the
altitude is in m, latitude in and logitude in rad, B is the magnetic field in T
in Cartesian coordinates, gmlat and gmlong the geomagnetic longitude and
latitude in rad, sim is a placeholder (empty list) for simulated data, and
item is a list of strings with the split items as described in
http://www.geomag.bgs.ac.uk/data_service/data/pmfformat.html
A file (sdatextr.html by default) with data can be obtained from
http://www.geomag.bgs.ac.uk/data_service/data/surveydata.shtml
For conviniece, indices ialt, ilat, ilong, iB, igmlat, igmlong, isim, iitem are given.

Example:
import geo
db = geo.db2(2013)
rec = db[0]
print(len(db), 'records retrieved', '\\n',
      'altitude =', rec[ialt], '\\n',
      'latitude =', rec[ilat], '\\n',
      'longitude', rec[ilong], '\\n',
      'B [T] =', rec[iB], '\\n',
      'item =', rec[iitem])
This results in the following output:
133 records retrieved 
 altitude = 0.0 
 latitude = 58.497 
 longitude 355.333 
 B [T] = [  3.11408808e-06  -1.01559188e-07   5.05352680e-05] 
 item = ['Loch_Eriboll___', ' 2013.486', '   31.503', ' 355.333', ...
'''

import numpy as np
import numpy.linalg as la

re = 6.3710e6  # Erdradius im m
B0 = 21.295e-6 / np.cos(np.deg2rad(63.30))  # Source: Formelbuch S. 204 (Zürich 2006.5)
ialt, ilat, ilong, iB, igmlat, igmlong, sim, iitem = 0, 1, 2, 3, 4, 5, 6, 7

def cart2spher(xyz):
    '''Convert Cartesian (x,y,z) coordinates to spherical (r,theta,phi) coordinates.
    A numpy.array is returned. See formulary page 114 with changes
    as -180 < phi <= 180 when using '''
    r = la.norm(xyz)
    theta = np.arcsin(xyz[2]/r)
    return np.array([r, theta, np.arctan2(xyz[1], xyz[0])])

def spher2cart(r, bg, lg):
    '''Convert spherical (r,theta,phi) coordinates to Cartesian (x,y,z) coordinates.
    A numpy.array is returned. See formulary page 114.'''
    a = np.cos(bg)
    b = np.cos(lg)
    c = np.sin(bg)
    d = np.sin(lg)
    return np.array([r * np.cos(bg) * np.cos(lg),
                     r * np.cos(bg) * np.sin(lg),
                     r * np.sin(bg)])

def __magnet__(db, line, year=2010, mindist=0):
    '''Interpret a line from MAGNET format and convert the field to Cartesian coordinates.
    A list (see module description) with data newer than year is appended to db
    if the distance to all geometric points in db is greater than mindist.
    For db a list is expected.'''
    rad = 180 / np.pi
    deg = np.pi / 180
    # Columns that separate items in data
    cols = [0, 15, 24, 33, 41, 51, 59, 66, 74, 82, 90, 97, 102, 104, 108, 116, 123, 127, 132]
    # Do a sanity check for each line as many data are invalid.
    # Ignore invalid lines
    try:  # try except handles conversion and other errors
        if len(line) > 134 or len(line) < 129:
            return
        # Extract items from each line
        item = []
        for i in range(len(cols) - 1):
            item.append(line[cols[i]:cols[i + 1]])
        # convert some data from string to numeric
        dat = float(item[1])
        if dat < year:  # Select by date
            return
        
        lat = (90 - float(item[2])) * deg
        long = float(item[3]) * deg
        incl = float(item[5]) * deg
        hor = float(item[6]) * 1e-9
        nor = float(item[7]) * 1e-9
        east = float(item[8]) * 1e-9
        vert = float(item[9]) * 1e-9
        tot = float(item[10]) * 1e-9
        alti = float(item[11])

        # Convert magn. field to Cartesian coordinates
        # Unit vector in vertical, east, and north direction
        ev = spher2cart(1, lat, long)
        ee = np.array([-np.sin(long), np.cos(long), 0.0])
        en = np.cross(ev, ee)

        B = vert * ev + east * ee + nor * en
        delta = [int(1e10*abs(hor - np.sqrt(nor**2 + east**2))),
                 int(1e10*abs(tot - la.norm(B))),
                 int(1e4*abs(incl - np.arctan2(vert, hor)))]
        # Check data consistency
        maxerror = 5  # used as multiple of resolution
        if (abs(hor - np.sqrt(nor**2 + east**2)) > maxerror * 1e-9
            or abs(tot - la.norm(B)) > maxerror * 1e-9
            or abs(incl - np.arctan2(vert, hor)) > maxerror * 0.001
            or alti < -500):
            return

        # Add data to db2 if far enough from all other points
        if mindist > 0:
            for i in range(len(db)):
                rec = db[i]
                dist = la.norm(spher2cart(rec[ialt] + re, rec[ilat], rec[ilong])
                               - spher2cart(alti + re, lat, long))
                if dist < mindist:
                    # If station is very close and data are newer, replace record
                    if (dist < mindist / 100) and (dat > float(rec[iitem][1])):
                        db[i] = [alti, lat, long, B, None, None, [], item]
                    return
        db.append([alti, lat, long, B, None, None, [], item])

    except ValueError:
        return  # If any conversion error occurs, the line is ignored

itemdesc = 'Station_name  Date       Colat  E-long    Declin   Inclin  Horiz   North  East    Vertic Total  Alt D so  SerNr   el_cod GMT Count'.split()


def db2(year=2010, mindist=0, file='sdatextr.html'):
    '''db2(year) is a data set that contains records that is newer
     than year (2010 is the default). To obtain a more equal
     distribution of points across the globe, a minimum distance
     between points can be defined. Data are taken from file.'''

    db = []
    with open(file) as f:
        for rec in f:
            __magnet__(db, rec, year, mindist)
    if len(db) == 0:
        print('Warning geo.db2: database is empty')
    return db

def load(fn):
    '''Load gedate from fn (pickle-file)'''
    import pickle
    with open(fn, 'rb') as f:
        return pickle.load(f)

    
# Format description for world-wide magnetic survey data
# from: http://www.geomag.bgs.ac.uk/data_service/data/pmfformat.html (with minor complements)

#nr Item	Characters	Fortran format	Description
# 0 A	1-15	A15	Station name/track
# 1 B	16-24	F9.3	Date
# 2 C	25-33	F9.3	Colatitude
# 3 D	34-41	F8.3	East longitude
# 4 E	42-51	F10.3	Declination
# 5 F	52-59	F8.3	Inclination
# 6 G	60-66	F7.0	Horizontal intensity
# 7 H	67-74	F8.0	North component
# 8 I	75-82	F8.0	East component
# 9 J	83-90	F8.0	Vertical component
#10 K	91-97	F7.0	Total intensity
#11 L	98-102	I5	Altitude
#12 M	103-104	I2	Data code
#13 N	105-108	I4	Source
#14 O	109-116	I8	Serial
#15 P	117-123	I7	Element code
#16 Q	124-127	I4	GMT
#17 R	128-132	A5	Country
#Notes
#Item B - date is in units of years with precision of 0.001 year. For example 7 April 2008 = 2008.266.
#Items C and D - colatitude (90 – latitude) and longitude are in units of degree with precision of 0.001 degree.
#Items E and F - declination and inclination are in units of degree with precision of 0.001 degree. Missing value code is 999.999 but other codes have been used in the past, eg. blanks, 0., 99.999, 999.000.
#Items G to K - the other elements are in units of nanoTeslas. Missing value code is 99999. but other codes have been used in the past, eg. blanks, 0., 999999., 88888.(?).
#Item L - altitude is in units of tens of metres. Missing value code is -999 but 0 has been used in the past
#Item M - data code. 1 - one-off land survey data, 2 - aeromagnetic, 4 - three-component marine data, 5 - satellite data, 6 - marine total intensity data, 9 - repeat station data, 0 - observatory annual means not in observatory annual means file. An asterisk in front of the data code indicates that the record has been changed.
#Item N - source number, as listed in file of sources. This is a list of references and letters.
#Item O - serial number. The system now used is to start at 1 for each new source.
#Item P - one digit for each of the 7 elements E to K. 2 - observed data, 9 - anomalous observed data (more than 1000 nT different from the International Geomagnetic Reference Field (IGRF)), 1 - computed data, 8 - anomalous computed data (more than 1000 nT different from IGRF) - this was introduced in March 1994, 0 - no data.
#Item Q – Greenwich Mean Time (GMT) usually given for codes 2, 5 and 6. However if GMT = 0 for data codes (Item M) 1 or 9 it indicates that the data have been reduced to a quiet night-time (LOCAL!) value.
#Item R - abbreviated country or state name or quad index (?).


#</p><pre><font face="Courier New, Courier, mono">
# A               B          C      D         E        F       G       H      I       J      K      L   MMNNNNOOOOOOOOPPPPPPPQQQQRRRRR
# 0               1          2      3         4        5       6       7      8       9      10     11  1413  14      15     16  17
# Station_name    Date       Colat  E-long    Declin   Inclin  Horiz   North  East    Vertic Total  Alt D so  SerNr   el_cod GMT Country
# VIVERO_2_______ 2015.500   46.304 352.391    -2.317  58.948 23595.  23576.   -954.  39188. 45743.   50 91145      122121121   0ES
# ESLES__________ 2015.500   46.717 356.219    -1.153  58.503 23970.  23965.   -482.  39120. 45879.  597 91145       42121121   0ES
# CASTRO_________ 2015.500   46.990 351.251    -2.864  58.006 24154.  24124.  -1207.  38664. 45589.  456 91145       32121121   0ES

##        delta = [int(1e10*abs(hor - np.sqrt(nor**2 + east**2))),
##                 int(1e10*abs(tot - la.norm(B))),
##                 int(1e4*abs(incl - np.arctan(vert/hor)*rad))]

