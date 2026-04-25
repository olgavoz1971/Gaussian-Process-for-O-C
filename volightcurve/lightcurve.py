import re

import astropy
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.io.votable import is_votable
from astropy.io import ascii
from astropy.units import UnitBase

from gavo.votable import votparse

import logging

logger = logging.getLogger(__name__)


class PhotCal:
    def __init__(self, zp_flux=1.0, zp_mag=0.0, mag_sys="Vega", filter_id=""):
        self.zp_flux = zp_flux  # photDM:PhotCal.zeroPoint.flux.value, Jy
        self.zp_mag = zp_mag  # photDM:PhotCal.zeroPoint.referenceMagnitude.value
        self.mag_sys = mag_sys  # photDM:PhotCal.magnitudeSystem.type
        # add stuff like ZeroPoint.type ({0=Pogson, 1=Asinh, 2=Linear} ), units etc .......
        self.filter_id = filter_id  # photDM:PhotometryFilter.identifier

    def mag_to_abs_flux(self, mag_array):
        return self.zp_flux * 10 ** (-0.4 * (mag_array - self.zp_mag))

    def mag_to_inst_flux(self, mag_array):
        return 1.0 * 10 ** (-0.4 * (mag_array - self.zp_mag))

    def flux_to_mag(self, flux_array, unit: UnitBase):
        if unit.is_equivalent(u.Jy):
            zp_flux = self.zp_flux
        else:
            zp_flux = 1.0       # we deal with instrumental flux
        return self.zp_mag - 2.5 * np.log10(flux_array / zp_flux)

    def __repr__(self):
        return (f"<PhotCal zeroPoint.referenceMagnitude={self.zp_mag}: zeroPoint.flux={self.zp_flux} Jy "
                f"filter_id={self.filter_id}>")


# class CooSys:


class TimeSys:
    def __init__(self, refposition='HELIOCENTER', timeorigin=0.0, timescale='UTC'):
        self.refposition = refposition  # HELIOCENTER OR BARYCENTER ...
        self.timeorigin = timeorigin  # JD0, f.e. 2400000.5
        self.timescale = timescale  # UTC, TCB, TBD etc

    @property
    def jd0(self):
        return self.timeorigin

    def __repr__(self):
        return f"<CooSys: timescale={self.timescale} refposition={self.refposition} timeorigin={self.timeorigin}>"


def extract_photcal(tree):
    """GAVO tree walker for PhotCal groups."""
    id_map = {}

    def map_ids(node, text, attrs, childIter):
        node_id = getattr(node, "id", None) or getattr(node, "ID", None)
        if node_id: id_map[node_id] = node
        for child in childIter:
            if hasattr(child, 'apply'): child.apply(map_ids)

    tree.apply(map_ids)

    calibrations = {}
    UT_FLUX = "photDM:PhotCal.zeroPoint.flux.value"
    UT_MAG = "photDM:PhotCal.zeroPoint.referenceMagnitude.value"
    UT_FILTER = "photDM:PhotometryFilter.identifier"

    # todo: extend this list

    def process_group(node, text, attrs, childIter):
        if node.name_ == 'GROUP' and getattr(node, "name", None) == "photcal":
            params = {'zp_flux': 1.0, 'zp_mag': 0.0, 'filter_id': ''}
            target_col = None
            for child in childIter:
                # Logic for PARAM and PARAMref
                target_param = None
                if child.name_ == 'PARAM':
                    target_param = child
                elif child.name_ == 'PARAMref':
                    target_param = id_map.get(getattr(child, "ref", None))
                elif child.name_ == 'FIELDref':
                    target_col = getattr(child, "ref", None)

                if target_param:
                    ut = getattr(target_param, "utype", "").lower()
                    if ut == UT_FLUX.lower():
                        params['zp_flux'] = float(target_param.value)
                    elif ut == UT_MAG.lower():
                        params['zp_mag'] = float(target_param.value)
                    elif ut == UT_FILTER.lower():
                        params['filter_id'] = target_param.value

            if target_col:
                calibrations[target_col] = PhotCal(**params)

        for child in childIter:
            if hasattr(child, 'apply'): child.apply(process_group)

    tree.apply(process_group)
    return calibrations


def extract_timesys(tree):
    """GAVO tree walker for TIMESYS."""
    data = {'timeorigin': 0.0, 'timescale': 'UTC', 'refposition': 'UNKNOWN'}

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


def get_error_colnames(table, base_ucd=None):
    """Finds error columns. If base_ucd provided (e.g. phot.mag), finds specific errors."""
    errors = find_columns_by_ucd(table, 'stat.error')
    if base_ucd:
        return [c for c in errors if base_ucd in table[c].info.meta.get('ucd', '')]
    return errors


def _promote_to_vo_standards(table):
    """Heuristically assign UCD/Units based on names (NO RENAMING)."""
    print(table.colnames)
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
        self.photcals = {}  # Map: column_name -> PhotCal instance

        self._ingest(file_path)

    def _ingest(self, file_path):
        """Main ingestion flow."""
        try:
            if is_votable(file_path):
                self.table = Table.read(file_path, format='votable')
                with open(file_path, "rb") as f:
                    votable_tree = votparse.readRaw(f)

                # Rigid Extraction
                self.timesys = extract_timesys(votable_tree)
                self.photcals = extract_photcal(votable_tree)
            else:
                self.table = Table.read(file_path)
                self.timesys.timeorigin = _pickup_jd0_from_table(self.table)

        except Exception as e:
            logger.warning(f"Standard read failed, trying heuristic ASCII...")
            # print(f"Standard read failed ({e}), trying heuristic ASCII...")
            self.table = ascii.read(file_path)
            self.table = _recover_lc_colnames(self.table)
            self.timesys.timeorigin = _pickup_jd0_from_table(self.table)

        # Post-process: Tag columns with UCDs/Units (No Renaming!)
        self.table = _promote_to_vo_standards(self.table)

        # Ensure we have photcal objects for every mag/flux column (even if dummy)
        self._fill_missing_calibrations()

        # # Update table metadata for storage
        # self.table.meta['jd0'] = self.timesys.jd0
        # self.table.meta['timesys'] = vars(self.timesys)

    def _fill_missing_calibrations(self):
        """Ensure every column tagged as mag/flux has a PhotCal instance."""
        for colname in self.table.colnames:
            ucd = self.table[colname].info.meta.get('ucd', '')
            if ('phot.mag' in ucd or 'phot.flux' in ucd) and 'stat.error' not in ucd and colname not in self.photcals:
                self.photcals[colname] = PhotCal()

    def __repr__(self):
        return f"<VOLightCurve: {len(self.table)} rows, {len(self.photcals)} PhotCals>"

    # for lazy people like me:

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

#
# # ------------------
#     def _inject_calibrated_column(self, source_col, new_col_name, transform_func, unit, ucd):
#         if source_col not in self.photcals:
#             raise ValueError(f"No PhotCal instance associated with column '{source_col}'")
#
#         cal_tool = self.photcals[source_col]
#
#         # Perform the actual math (transform_func is passed as an argument)
#         transformed_data = transform_func(self.table[source_col])
#
#         # Add to internal table
#         self.table[new_col_name] = transformed_data
#         self.table[new_col_name].unit = unit
#         self.table[new_col_name].info.meta = {'ucd': ucd}
#
#         # Link the new column to the same calibration tool
#         self.photcals[new_col_name] = cal_tool
#         return new_col_name
#
#     def add_abs_flux_column_from_mag(self, mag_col, new_col_name='abs_flux_from_mag'):
#         return self._inject_calibrated_column(
#             source_col=mag_col,
#             new_col_name=new_col_name,
#             transform_func=lambda col: self.photcals[mag_col].mag_to_abs_flux(col.data),
#             unit=u.Jy,
#             ucd='phot.flux.density'
#         )
#
#     def add_inst_flux_column_from_mag(self, mag_col, new_col_name='inst_flux_from_mag'):
#         return self._inject_calibrated_column(
#             source_col=mag_col,
#             new_col_name=new_col_name,
#             transform_func=lambda col: self.photcals[mag_col].mag_to_inst_flux(col.data),
#             unit=(1 / u.s),
#             ucd='phot.flux'
#         )
#
#     def add_mag_column_from_flux(self, flux_col, new_col_name='mag_from_flux'):
#         return self._inject_calibrated_column(
#             source_col=flux_col,
#             new_col_name=new_col_name,
#             transform_func=lambda col: self.photcals[flux_col].flux_to_mag(col.data, col.unit),
#             unit=u.mag,
#             ucd='phot.mag;stat.filled'
#         )

# -----------
    def add_abs_flux_column_from_mag(self, mag_col, new_col_name='abs_flux_from_mag'):
        """
        Takes a magnitude column, finds its PhotCal,
        and adds a new absolute flux column (Jy).
        """
        if mag_col not in self.photcals:
            raise ValueError(f"No PhotCal instance associated with column '{mag_col}'")

        cal_tool = self.photcals[mag_col]
        flux_data = cal_tool.mag_to_abs_flux(self.table[mag_col].data)

        col_out = new_col_name or f"{mag_col}_flux_jy"

        # Add to internal table
        self.table[col_out] = flux_data
        self.table[col_out].unit = u.Jy
        self.table[col_out].info.meta = {'ucd': 'phot.flux.density'}

        # Link the same PhotCal object to the new column as well
        self.photcals[col_out] = cal_tool
        return col_out

    def add_inst_flux_column_from_mag(self, mag_col, new_col_name='inst_flux_from_mag'):
        """
        Takes a magnitude column, finds its PhotCal,
        and adds a new instrumental flux column (1/s).
        """
        if mag_col not in self.photcals:
            raise ValueError(f"No PhotCal instance associated with column '{mag_col}'")

        cal_tool = self.photcals[mag_col]
        flux_data = cal_tool.mag_to_inst_flux(self.table[mag_col].data)

        col_out = new_col_name or f"{mag_col}_flux"

        # Add to internal table
        self.table[col_out] = flux_data
        self.table[col_out].unit = (1 / u.s)
        self.table[col_out].info.meta = {'ucd': 'phot.flux'}

        # Link the same PhotCal object to the new column as well
        self.photcals[col_out] = cal_tool
        return col_out

    def add_mag_column_from_flux(self, flux_col, new_col_name='mag_from_flux'):
        if flux_col not in self.photcals:
            raise ValueError(f"No PhotCal instance associated with column '{flux_col}'")

        cal_tool = self.photcals[flux_col]
        mag_data = cal_tool.flux_to_mag(self.table[flux_col].data, self.table[flux_col].unit)

        col_out = new_col_name or f"{flux_col}_mag"

        self.table[col_out] = mag_data
        self.table[col_out].unit = u.mag
        self.table[col_out].info.meta = {'ucd': 'phot.mag;stat.filled'}

        self.photcals[col_out] = cal_tool
        return col_out
# ---------------

    def get_time_colnames(self): return get_time_colnames(self.table)
    def get_mag_colnames(self): return get_mag_colnames(self.table)
    def get_flux_colnames(self): return get_flux_colnames(self.table)


def print_col_ucd(lc: VOLightCurve):
    for colname in lc.colnames:
        print(f"Col: {colname:10} Unit: {lc[colname].unit}"
              f"UCD: {lc[colname].info.meta.get('ucd', 'None')}")


def main():
    # for filename in ['../data/my_g3.vot', '../data/AY_Lac-R.vot']:
    #     with open(filename, "rb") as f:
    #         votable_tree = votparse.readRaw(f)
    #     res = extract_photcal(votable_tree)
    #     print(res)

    for filename in [
        # '../data/lc_tess_HD182144_TIC_406949643_sector__40_author__SPOC_methods__pdcsap.vot',
        # '../data/OGLE-SMC-CEP-0325-I.vot',
        # '../data/6009363278148078848-G.vot',
        #
        # '../data/AY_Lac-R.vot',
        '../data/g2_jk.vot',
        '../data/my_g3.vot',
        '../data/ASAS19pm/ASas19pm.dat'
    ]:
        print(f'Ingesting {filename}')
        lc = VOLightCurve(file_path=filename)
        print_col_ucd(lc)
        print(f'{lc.photcals=}')
        print(f'{lc.timesys=}\n')
        print('time columns:', get_time_colnames(lc))
        print('flux columns:', get_flux_colnames(lc))
        print('mag columns:', get_mag_colnames(lc))
        for colname in lc.get_flux_colnames():
            lc.add_mag_column_from_flux(colname)
            print(lc['mag_from_flux', colname][0])
        for colname in lc.get_mag_colnames():
            lc.add_inst_flux_column_from_mag(colname, colname+'_inst_flux')
            print(lc[colname+'_inst_flux', colname][0])
        # lc.add_inst_flux_column_from_mag('phot')
        # lc.add_abs_flux_column_from_mag('phot')
        # print(lc['flux', 'abs_flux_from_mag'][0])
        # pass


if __name__ == "__main__":
    main()
