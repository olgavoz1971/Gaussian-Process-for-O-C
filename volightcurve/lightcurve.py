import re
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.table import MaskedColumn
from astropy.io.votable import is_votable
from astropy.io import ascii

from gavo.votable import votparse

import logging

logger = logging.getLogger(__name__)


class PhotometryFilter:
    def __init__(self, filter_id=None, spectral_location=None, spectral_location_unit: str = None):
        self._filter_id = filter_id  # photDM:PhotometryFilter.identifier

        # Build the physical quantity
        if spectral_location is None:
            self.spectral_location = None
        else:
            if isinstance(spectral_location, u.Quantity):
                self.spectral_location = spectral_location
            else:
                # Pair the value with the unit (defaulting to dimensionless if None)
                unit = u.dimensionless_unscaled
                if spectral_location_unit is not None:
                    try:
                        unit = u.Unit(spectral_location_unit)
                    except Exception as e:
                        logger.warning(f'Inappropriate spectral_location_unit {spectral_location_unit}: {type(e)}')

                self.spectral_location = spectral_location * unit
        # todo: blah-blah-blah

    @property
    def filter_id(self):
        return self._filter_id

    @filter_id.setter
    def filter_id(self, value):
        if not isinstance(value, str):
            raise ValueError("Filter ID must be a string.")
        self._filter_id = value.strip()

    def __repr__(self):
        return (f"<PhotometryFilter: filter_id={self.filter_id} "
                f"spectralLocation={self.spectral_location}>")


class PhotCal:
    def __init__(self, zp_flux=1.0, zp_flux_unit=None, zp_mag=0.0, zp_mag_unit=None, mag_sys="Vega"):
        # zp_flux, unit-aware. # photDM:PhotCal.zeroPoint.flux.value
        if isinstance(zp_flux, u.Quantity):
            self._zp_flux = zp_flux
        else:
            unit = u.dimensionless_unscaled
            if zp_flux_unit:
                try:
                    unit = u.Unit(zp_flux_unit)
                except Exception as e:
                    logger.warning(f"Invalid zp_flux_unit '{zp_flux_unit}': {type(e)}")
            self._zp_flux = zp_flux * unit

        # zp_mag  photDM:PhotCal.zeroPoint.referenceMagnitude.value
        # Here. I hope, I can guess the default ;-)
        if isinstance(zp_mag, u.Quantity):
            self._zp_mag = zp_mag
        else:
            m_unit = u.mag
            if zp_mag_unit:
                try:
                    m_unit = u.Unit(zp_mag_unit)
                except Exception as e:
                    logger.warning(f"Invalid zp_mag_unit '{zp_mag_unit}': {type(e)}")
            self._zp_mag = zp_mag * m_unit

        self._mag_sys = mag_sys  # photDM:PhotCal.magnitudeSystem.type

        # todo: add stuff like ZeroPoint.type ({0=Pogson, 1=Asinh, 2=Linear} ), units etc .......

    @property
    def zp_flux(self):
        return self._zp_flux

    @zp_flux.setter
    def zp_flux(self, value):
        if not isinstance(value, (int, float, np.number)):
            raise ValueError("Zero point flux must be a number.")
        self._zp_flux = float(value)

    @property
    def zp_mag(self):
        return self._zp_mag

    @zp_mag.setter
    def zp_mag(self, value):
        if not isinstance(value, (int, float, np.number)):
            raise ValueError("Zero point magnitude must be a number.")
        self._zp_mag = float(value)

    @property
    def mag_sys(self):
        return self._mag_sys

    @mag_sys.setter
    def mag_sys(self, value):
        # Example validation: ensure it's a string and capitalized
        if not isinstance(value, str):
            raise ValueError("Magnitude system must be a string.")
        self._mag_sys = value.strip()

    def mag_to_flux(self, mag):
        """I hope, flux and zp_flux are astropy quantities"""
        if mag.unit is None or not mag.unit.is_equivalent(self._zp_mag.unit):
            raise u.UnitsError(f'flux column unit[{mag.unit}] and zero_point unit[{self._zp_mag.unit}] must match')
        return self.zp_flux * 10 ** (-0.4 * (mag - self.zp_mag).value)  # oh!

    def flux_to_mag(self, flux):
        """I hope, flux and zp_flux are astropy quantities"""
        if flux.unit is None or not flux.unit.is_equivalent(self._zp_flux.unit):
            raise u.UnitsError(f'flux column unit[{flux.unit}] and zero_point unit[{self._zp_flux.unit}] must match')
        ratio = (flux / self._zp_flux).value
        return self.zp_mag - 2.5 * np.log10(ratio) * u.mag  # oh!

    def __repr__(self):
        return (f"<PhotCal zeroPoint.referenceMagnitude={self.zp_mag}: "
                f"zeroPoint.flux={self.zp_flux} "
                f"magnitudeSystem={self.mag_sys}>")


class PhotDM:
    def __init__(self, photcal: PhotCal = None, photometry_filter: PhotometryFilter = None):
        self.photcal = photcal
        self.filter = photometry_filter
        # todo: blah-blah-blah

    @property
    def filter_id(self):
        """Shortcut to get the filter name without digging into the sub-object."""
        return self.filter.filter_id if self.filter else "Unknown"

    @filter_id.setter
    def filter_id(self, value):
        self.filter.filter_id = value

    @property
    def mag0(self):
        """Shortcut to get the filter name without digging into the sub-object."""
        return self.photcal.zp_mag if self.photcal else 0.0

    @mag0.setter
    def mag0(self, value):
        self.photcal.zp_mag = value

    def __repr__(self):
        # f_id = self.filter_id
        # sys = self.photcal.mag_sys if self.photcal else "None"
        # return f"<PhotDM Filter={f_id} Sys={sys}>"
        return f"<PhotDM Filter={self.filter} photcal={self.photcal}>"


class CooSys:
    def __init__(self, epoch=2016, system='ICRS'):
        self.epoch = epoch
        self.system = system

    def __repr__(self):
        return f"<CooSys epoch={self.epoch}: system={self.system}>"


class TimeSys:
    def __init__(self, refposition='HELIOCENTER', timeorigin=0.0, timescale='UTC'):
        self.refposition = refposition  # HELIOCENTER OR BARYCENTER ...
        self.timeorigin = timeorigin  # JD0, f.e. 2400000.5
        self.timescale = timescale  # UTC, TCB, TBD etc

    @property
    def jd0(self):
        return self.timeorigin

    def __repr__(self):
        return f"<TimeSys: timescale={self.timescale} refposition={self.refposition} timeorigin={self.timeorigin}>"


def extract_photdm(tree):
    """GAVO tree walker for PhotCal groups
    Note: This is DACHS phot-0 (with my adding) -specific
    """
    id_map = {}

    def map_ids(node, text, attrs, childIter):
        node_id = getattr(node, "id", None) or getattr(node, "ID", None)
        if node_id: id_map[node_id] = node
        for child in childIter:
            if hasattr(child, 'apply'): child.apply(map_ids)

    tree.apply(map_ids)

    dm_map = {}

    UT_FLUX = "photDM:PhotCal.zeroPoint.flux.value"
    UT_MAG = "photDM:PhotCal.zeroPoint.referenceMagnitude.value"
    UT_MAG_SYS = "photDM:PhotCal.magnitudeSystem.type"
    UT_FILTER = "photDM:PhotometryFilter.identifier"
    UT_FILTER_SPEC = "photDM:PhotometryFilter.spectralLocation.value"

    # todo: extend this list

    def process_group(node, text, attrs, childIter):
        if node.name_ == 'GROUP' and getattr(node, "name", None) == "photcal":
            # arguments of PhotCal.__init__
            cal_params = {'zp_flux': 1.0, 'zp_mag': 0.0, 'zp_mag_unit': 'mag', 'zp_flux_unit': None}
            # these are arguments of the PhotomeryPilter.__init__
            filter_params = {'filter_id': '', 'spectral_location': 0.0, 'spectral_location_unit': None}
            # filter_id = ''
            # filter_spec = None
            # filter_spec_unit = None
            target_col = None
            for child in childIter:
                # Logic for PARAM and PARAMref
                target_param = None
                # Note: we should always look for the utype on the child (the reference) first!
                # The referenced parameter may have another utype
                role_utype = getattr(child, "utype", "").lower()

                if child.name_ == 'PARAM':
                    target_param = child
                elif child.name_ == 'PARAMref':
                    target_param = id_map.get(getattr(child, "ref", None))
                elif child.name_ == 'FIELDref':
                    target_col = getattr(child, "ref", None)

                if target_param is not None:
                    # If the reference (child) didn't have utype,
                    # use the parameter's own utype (so, I do not like this case)
                    ut = role_utype or getattr(target_param, "utype", "").lower()

                    if ut == UT_FLUX.lower():
                        cal_params['zp_flux'] = float(target_param.value)
                        cal_params['zp_flux_unit'] = getattr(target_param, "unit", None)
                    elif ut == UT_MAG.lower():
                        cal_params['zp_mag'] = float(target_param.value)
                        cal_params['zp_mag_unit'] = target_param.unit
                    elif ut == UT_MAG_SYS.lower():
                        cal_params['mag_sys'] = target_param.value
                    elif ut == UT_FILTER.lower():
                        filter_params['filter_id'] = target_param.value
                    elif ut == UT_FILTER_SPEC.lower():
                        filter_params['spectral_location'] = float(target_param.value)
                        filter_params['spectral_location_unit'] = getattr(target_param, "unit", None)

            if target_col:
                phot_filter = PhotometryFilter(**filter_params)
                # filter_id=filter_id, spectral_location=filter_spec, spectral_location_unit=filter_spec_unit)
                photcal = PhotCal(**cal_params)
                # calibrations[target_col] = PhotCal(**params)
                # Assign the DM hub to the column name
                dm_map[target_col] = PhotDM(photcal=photcal, photometry_filter=phot_filter)

        # Recursive walk
        for child in childIter:
            if hasattr(child, 'apply'): child.apply(process_group)

    tree.apply(process_group)
    return dm_map


def extract_timesys(tree):
    """GAVO tree walker for TIMESYS."""
    data = {'timeorigin': 0.0, 'timescale': 'UTC', 'refposition': 'HELIOCENTER'}

    def find_ts(node, text, attrs, childIter):
        if node.name_ == 'TIMESYS':
            data['timeorigin'] = float(attrs.get('timeorigin', 0.0))
            data['timescale'] = attrs.get('timescale', 'UTC')
            data['refposition'] = attrs.get('refposition', 'UNKNOWN')
        for child in childIter:
            if hasattr(child, 'apply'): child.apply(find_ts)

    tree.apply(find_ts)
    return TimeSys(**data)


def find_columns_by_ucd(table, ucd_fragment):
    """Returns a list of all column names containing the UCD fragment."""
    matches = []
    for colname in table.colnames:
        col_ucd = table[colname].info.meta.get('ucd', '') if table[colname].info.meta else ''
        if ucd_fragment in col_ucd:
            matches.append(colname)
    return matches


def get_time_colnames(table):
    return find_columns_by_ucd(table, 'time.epoch')


def get_mag_colnames(table):
    # Returns primary magnitudes, excludes errors
    all_mags = find_columns_by_ucd(table, 'phot.mag')
    return [c for c in all_mags if 'stat.error' not in table[c].info.meta.get('ucd', '')]


def get_flux_colnames(table):
    all_flux = find_columns_by_ucd(table, 'phot.flux')
    return [c for c in all_flux if 'stat.error' not in table[c].info.meta.get('ucd', '')]


def is_mag_column(table: Table, colname: str | None):
    if colname is None: return False
    return 'phot.mag' in table[colname].info.meta.get('ucd', '')


def is_flux_column(table: Table, colname: str | None):
    if colname is None: return False
    return 'phot.flux' in table[colname].info.meta.get('ucd', '')


def get_error_colnames(table, base_ucd=None):
    """Finds error columns. If base_ucd provided (e.g. phot.mag), finds specific errors."""
    errors = find_columns_by_ucd(table, 'stat.error')
    if base_ucd:
        return [c for c in errors if base_ucd in table[c].info.meta.get('ucd', '')]
    return errors


def _promote_to_vo_standards(table):
    """Heuristically assign UCD/Units based on names (NO RENAMING)."""
    for colname in table.colnames:
        col = table[colname]
        name_low = colname.lower()
        if not col.unit:
            if 'mag' in name_low:
                col.unit = u.mag
            elif 'flux' in name_low:
                col.unit = u.Jy  # default assumption
            elif any(k in name_low for k in ['time', 'jd', 'mjd']):
                col.unit = u.d

        if not col.info.meta or not col.info.meta.get('ucd'):
            if col.info.meta is None: col.info.meta = {}
            if 'mag' in name_low:
                ucd = 'phot.mag'
            elif 'flux' in name_low:
                ucd = 'phot.flux'
            elif any(k in name_low for k in ['time', 'jd', 'mjd']):
                ucd = 'time.epoch'
            else:
                continue

            if any(k in name_low for k in ['err', 'uncert', 'sigma']):
                ucd = f"stat.error;{ucd}"
            col.info.meta['ucd'] = ucd
    return table


def _pickup_jd0_from_table(table):
    jd0_pattern = re.compile(r"JD0\s*=\s*([+-]?\d*\.?\d+)")
    for line in table.meta.get('comments', []):
        match = jd0_pattern.search(line.upper())
        if match: return float(match.group(1))
    return 0.0


def _pickup_mag0_from_table(table):
    jd0_pattern = re.compile(r"MAG0\s*=\s*([+-]?\d*\.?\d+)")
    for line in table.meta.get('comments', []):
        match = jd0_pattern.search(line.upper())
        if match: return float(match.group(1))
    return 0.0


def _pickup_filter_from_table(table):
    """
    Scans the table comments for filter/band identification.
    Matches patterns like: FILTER=Gaia_G.v2 or BAND = r
    """
    # Matches alphanumeric, dots, underscores, and dashes after the '='
    filter_pattern = re.compile(r"(?:FILTER|BAND)\s*=\s*([\w\.\-]+)")

    for line in table.meta.get('comments', []):
        match = filter_pattern.search(line.upper())
        if match:
            # We return the original case from the line, not the upper() version
            # so 'Gaia_G' doesn't become 'GAIA_G'
            actual_line = line.split('=')[-1].strip()
            return actual_line.split()[0]  # Take first word to avoid trailing comments

    return "Unknown"


def _recover_lc_colnames(table):
    """
    Strictly renames columns for 'colN' tables based on comments or position.
    Requirement: A comment line must have exactly the same number of words
    as the table has columns to be considered a valid header.
    """
    comments = table.meta.get('comments', [])
    num_cols = len(table.colnames)
    found_header = None

    # Look into the comments
    for line in comments:
        # Remove #, strip, and split into words
        parts = line.strip().lstrip('#').strip().split()

        # STRICT CHECK: Word count must equal Column count
        if len(parts) == num_cols:
            found_header = parts
            break

    # Apply names
    for i, colname in enumerate(table.colnames):
        if found_header:
            new_name = found_header[i]
        else:
            # RIGID POSITIONAL FALLBACK
            if i == 0:
                new_name = 'obs_time'
            elif i == 1:
                new_name = 'mag'
            elif i == 2:
                new_name = 'mag_err'
            else:
                new_name = f'col{i + 1}'

        if colname != new_name:
            table.rename_column(colname, new_name)
    return table


class VOLightCurve:
    def __init__(self, file_path):
        self.file_path = file_path
        self.table = None
        self.timesys = TimeSys()
        self.coosys = CooSys()
        self.photdms = {}  # maps column_name -> PhotDM instance

        self._ingest(file_path)

    def _ingest(self, file_path):
        """Main ingestion flow."""
        try:
            if is_votable(file_path):
                # The best track:
                self.table = Table.read(file_path, format='votable')
                if hasattr(file_path, 'read'):
                    # It's already an open stream! (from the Dash Plotly Upload for instance
                    # Just ensure we are at the start of the "file"
                    file_path.seek(0)
                    votable_tree = votparse.readRaw(file_path)
                else:
                    # It's a string path, we need to open it
                    with open(file_path, "rb") as f:
                        votable_tree = votparse.readRaw(f)

                # and yet I don't trust you:
                self.table = _promote_to_vo_standards(self.table)

                # Rigid Extraction
                self.timesys = extract_timesys(votable_tree)
                self.photdms = extract_photdm(votable_tree)
                return

            # Try to fix things ...
            self.table = Table.read(file_path)
            logger.info(f"We have read {file_path} using Table.read, though it is not the right VOTable")

        except Exception as e:
            logger.warning(f"Standard read failed, trying heuristic ASCII...")
            # print(f"Standard read failed ({e}), trying heuristic ASCII...")
            self.table = ascii.read(file_path)
            self.table = _recover_lc_colnames(self.table)

        # Post-process: Tag columns with UCDs/Units
        self.table = _promote_to_vo_standards(self.table)
        # Try to extract metadata
        self.timesys.timeorigin = _pickup_jd0_from_table(self.table)
        heur_filter_id = _pickup_filter_from_table(self.table)
        heur_mag0 = _pickup_mag0_from_table(self.table)

        # put this filter into the photcal and attach photCal to any mag/flux columns
        for colname in self.get_flux_colnames() + self.get_mag_colnames():
            photdm = self.photdms.get(colname, None)

            # todo: sanitise this defaults/zeros/None logic
            if photdm is None:
                new_filter = PhotometryFilter(filter_id=heur_filter_id)
                if heur_mag0:
                    photcal = PhotCal(zp_mag=heur_mag0, zp_mag_unit='mag')
                else:
                    photcal = None
                self.photdms[colname] = PhotDM(photcal=photcal, photometry_filter=new_filter)
            else:
                # If it already existed but had no filter name, update it
                if photdm.filter_id is None:
                    photdm.filter_id = heur_filter_id
                if heur_mag0 is not None:
                    photdm.mag0 = heur_mag0

        # Ensure we have photcal objects for every mag/flux column (even if dummy)
        # self._fill_missing_calibrations()

        # # Update table metadata for storage
        # self.table.meta['jd0'] = self.timesys.jd0
        # self.table.meta['timesys'] = vars(self.timesys)

    def __repr__(self):
        return f"<VOLightCurve: {len(self.table)} rows, {len(self.photdms)} PhotCals, jd0={self.jd0}>"

    def __getitem__(self, key):
        return self.table[key]

    def __getattr__(self, name):
        """
        Allows lc.colnames, lc.meta, lc.row_groups, etc.
        If the attribute isn't found in VOLightCurve,
        Python will look for it in self.table.
        """
        # Avoid infinite recursion if table isn't initialized yet
        if name == "table":
            raise AttributeError("Table not yet initialized")

        try:
            return getattr(self.table, name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __len__(self):
        """Allows len(lc) to return the number of rows"""
        return len(self.table)

    @property
    def jd0(self):
        # Yes, I realise that I ought to have a set of timesys connected to time columns,
        # but I'm too lazy to implement this (yet)
        return self.timesys.jd0

    def add_flux_column_from_mag(self, mag_col_name, new_col_name=None):
        """
        Takes a magnitude column, finds its PhotCal,
        and adds a new flux column. A unit of the new flux is specified by photCal zp_flux unit
        """

        colname_out = new_col_name or f"flux_from_{mag_col_name}"
        photdm = self.photdms.get(mag_col_name, None)

        if photdm is None or photdm.photcal is None:
            logger.warning(f'No PhotCal for {mag_col_name}. Filling {new_col_name} with None/NaN')
            flux = np.full(len(self.table), np.nan)
            self.table[new_col_name] = MaskedColumn(data=flux, mask=np.isnan(flux))
        else:
            # Unit safety:
            try:
                flux = photdm.photcal.mag_to_flux(self.table[mag_col_name])
            except (u.UnitConversionError, u.UnitTypeError, u.UnitsError, TypeError, ValueError) as e:
                logger.error(f"Unit conversion failed for {mag_col_name} [{self.table[mag_col_name].unit}]: {e}")
                data = np.full(len(self.table), np.nan)
                flux = MaskedColumn(data=data, mask=np.ones(len(data), dtype=bool))

            # Check physical equivalence: Flux Density is power/area/freq (or wavelength)
            ucd = 'phot.flux'
            if hasattr(flux, 'unit') and flux.unit is not None:
                # Check if it's a density (per Hz OR per AA/m/nm)
                u_flux_nu = u.W / (u.m ** 2 * u.Hz)  # Like Jy
                u_flux_lam = u.W / (u.m ** 2 * u.m)  # Like erg/(s*cm2*AA)
                if flux.unit.is_equivalent(u_flux_nu) or flux.unit.is_equivalent(u_flux_lam):
                    ucd = 'phot.flux.density'

            self.table[colname_out] = flux
            self.table[colname_out].info.meta = {'ucd': ucd}

            # Link the same DataModel
            self.photdms[colname_out] = photdm

        return colname_out

    def add_mag_column_from_flux(self, flux_col_name, new_col_name='mag_from_flux'):
        colname_out = new_col_name or f"mag_from_{flux_col_name}"
        photdm = self.photdms.get(flux_col_name, None)

        if photdm is None or photdm.photcal is None:
            logger.warning(f'No PhotCal for {flux_col_name}. Filling {new_col_name} with None/NaN')
            mag = np.full(len(self.table), np.nan)
            self.table[new_col_name] = MaskedColumn(data=mag, mask=np.isnan(mag))
        else:
            # Unit safety:
            try:
                mag = photdm.photcal.flux_to_mag(self.table[flux_col_name])
            except (u.UnitConversionError, u.UnitTypeError, u.UnitsError, TypeError, ValueError) as e:
                logger.error(
                    f"Unit conversion failed for {flux_col_name} [{self.table[flux_col_name].unit}]: {e}")
                data = np.full(len(self.table), np.nan)
                mag = MaskedColumn(data=data, mask=np.ones(len(data), dtype=bool))

            # Check physical equivalence: Flux Density is power/area/freq (or wavelength)
            ucd = 'phot.mag'
            self.table[colname_out] = mag
            self.table[colname_out].info.meta = {'ucd': ucd}

            # Link the same DataModel
            self.photdms[colname_out] = photdm

        return colname_out

    def get_time_colnames(self):
        return get_time_colnames(self.table)

    def get_mag_colnames(self):
        return get_mag_colnames(self.table)

    def get_flux_colnames(self):
        return get_flux_colnames(self.table)

    def get_mag_error_colnames(self):
        return get_error_colnames(self.table, base_ucd='phot.mag')

    def get_flux_error_colnames(self):
        return get_error_colnames(self.table, base_ucd='phot.flux')


def print_col_ucd(lc: VOLightCurve):
    for colname in lc.colnames:
        print(f"Col: {colname:10} Unit: {lc[colname].unit} "
              f"UCD: {lc[colname].info.meta.get('ucd', 'None')}")


def main():
    for filename in [
        # 'data/lc_tess_HD182144_TIC_406949643_sector__40_author__SPOC_methods__pdcsap.vot',
        'data/lc_tess_HD182144_TIC_406949643_sector__40_author__SPOC_methods__pdcsap.ecsv'
        # 'data/OGLE-SMC-CEP-0325-I.vot',
        # 'data/6009363278148078848-G.vot',
        # 'data/AY_Lac-R.vot',
        # 'data/g2_jk.vot',
        # 'data/my_g3.vot',
        # 'data/ASas19pm.dat'
    ]:
        print(f'\n\n\n Ingesting {filename}')
        lc = VOLightCurve(file_path=filename)
        print_col_ucd(lc)
        print(f'{lc.photdms=}')
        print(f'{lc.timesys=}\n')
        print('time columns:', get_time_colnames(lc))
        print('flux columns:', get_flux_colnames(lc))
        print('mag columns:', get_mag_colnames(lc))

        print(lc[0])
        print('\n\n\n All flux columns. Convert into magnitudes')
        for colname in lc.get_flux_colnames():
            out_colname = f'magnitude_from_{colname}'
            print(f'flux:{colname} --> mag:{out_colname}')
            lc.add_mag_column_from_flux(colname, out_colname)
            print(lc[out_colname, colname][0])
        print('\n\n\nAll magnitude columns. Convert into flux')
        for colname in lc.get_mag_colnames():
            out_colname = f'flux_from_{colname}'
            print(f'mag:{colname} --> flux:{out_colname}')
            lc.add_flux_column_from_mag(colname, out_colname)
            print(lc[out_colname, colname][0])


if __name__ == "__main__":
    main()
