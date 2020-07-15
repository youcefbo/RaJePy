import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates.angles import Longitude, Latitude, Angle


def elevation(coord, lat, lst):
    """
    Given a celestial coordinate and observatory latitude, calculate the
    celestial coordinate's elevation given an LST.

    Parameters
    ----------
    coord: astropy.coordinates.SkyCoord
        Object's RA/DEC
    lat: float
        Observatory latitude in degrees
    lst: float
        Local sideral time in hours

    Returns
    -------
    Elevation as an astropy.coordinates.angles.Angle instance in units of
    degrees

    """
    assert isinstance(coord, SkyCoord)
    assert isinstance(lat, float)
    assert isinstance(lst, float)

    if lst < 0. or lst > 24.:
        raise ValueError("lst must be in range 0 <= lst < 24, "
                         "not {}".format(lst))

    if lat < -90. or lat > 90.:
        raise ValueError("lat must be in range -90 <= lat <= +90,"
                         "not {}".format(lat))

    lat_angle = Latitude(lat, unit=u.deg)
    lst_angle = Longitude(lst, unit=u.hour)
    ra, dec = coord.ra, coord.dec
    ha_angle = ra - lst_angle

    p1 = np.sin(lat_angle.rad) * np.sin(dec.rad)
    p2 = np.cos(lat_angle.rad) * np.cos(dec.rad) * np.cos(ha_angle.rad)

    return Angle(np.arcsin(p1 + p2) * u.rad).deg


def ha(coord, lat, elev):
    """
    Given a celestial coordinate and observatory latitude, calculate the
    celestial coordinate's hour angle at a given elevation.

    Parameters
    ----------
    coord: astropy.coordinates.SkyCoord
        Object's RA/DEC
    lat: float
        Observatory latitude in degrees
    elev: float
        Elevation to compute hour angle at in degrees

    Returns
    -------
    Hour angle as an astropy.coordinates.angles.Longitude instance in units
    of hours

    """
    assert isinstance(coord, SkyCoord)
    assert isinstance(lat, float)
    assert isinstance(elev, float)

    if elev < -90. or elev > 90.:
        raise ValueError("elev must be in range -90 <= el <= +90, "
                         "not {}".format(elev))

    if lat < -90. or lat > 90.:
        raise ValueError("lat must be in range -90 <= lat <= +90,"
                         "not {}".format(lat))

    lat_angle = Latitude(lat, unit=u.deg)
    el_angle = Angle(elev * u.deg)
    ra, dec = coord.ra, coord.dec

    p1 = np.sin(el_angle.rad) - np.sin(lat_angle.rad) * np.sin(dec.rad)
    p2 = np.cos(lat_angle.rad) * np.cos(dec.rad)

    return Longitude(np.arccos(p1 / p2) * u.rad).hourangle

# Module testing code below
if __name__ == '__main__':
    tgt_c = SkyCoord("06:12:54.02", "19:59:23.6",
                     unit=(u.hourangle, u.deg), frame='fk5')
    tgt_c_s = tgt_c.to_string('hmsdms')

    lat = 34.
    lst = 4.0
    elev = 20.

    print("Test func el('{}', {}, {}): {:.3f}deg".format(tgt_c_s, lat, lst,
                                                         elevation(tgt_c, lat,
                                                                   lst)))
    print("Test func ha('{}', {}, {}): {:.3f}h".format(tgt_c_s, lat, elev,
                                                       ha(tgt_c, lat, elev)))

    import matplotlib.pylab as plt

    lsts = np.linspace(0, 24, 1000)
    els = [elevation(tgt_c, lat, _) for _ in lsts]

    plt.close('all')

    plt.plot(lsts, els, 'b-')

    min_el_lst1 = tgt_c.ra.hourangle + ha(tgt_c, lat, elev)
    min_el_lst2 = tgt_c.ra.hourangle - ha(tgt_c, lat, elev)

    if min_el_lst1 > 24.:
        min_el_lst1 = min_el_lst1 - 24.

    if min_el_lst2 < 0.:
        min_el_lst2 = min_el_lst2 + 24.

    min_el_lst = (min_el_lst1, min_el_lst2)

    ylims = plt.ylim()
    ylims = (ylims[0], 90.)

    plt.vlines(min_el_lst, ylims[0], ylims[1], colors='k',
               ls=':')

    min_el_lst_s = []
    for el in min_el_lst:
        if np.isnan(el):
            min_el_lst_s.append('')
        else:
            s = Longitude(el * u.hourangle).to_string(sep='hms', precision=0)
            min_el_lst_s.append(s)

    plt.annotate(min_el_lst_s[0], (min_el_lst[0], elev))
    plt.annotate(min_el_lst_s[1], (min_el_lst[1], elev))

    xlims = (0, 24)
    plt.xlim(xlims)

    plt.hlines(elev, xlims[0], xlims[1], color='k', ls=':')

    plt.ylim(ylims)
    plt.xlim(xlims)
    plt.xlabel('LST [hour]')
    plt.ylabel('Elevation [deg]')

    plt.show()