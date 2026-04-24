# https://www.ivoa.net/documents/VOTable/20230913/WD-VOTable-1.5-20230913.html
import re
from astropy.table import Table
from astropy.io.votable import is_votable
from astropy.io import ascii
import numpy as np
import astropy.units as u
from gavo.votable import votparse


zero_point_G2 = 25.6884
flux0_G2 = 2836.53


def get_mag_from_mag(flux, flux0=flux0_G2, mr=0):
    mag = mr - 2.5 * np.log10(flux / flux0)
    return mag


def get_flux_from_mag(mag, flux0=flux0_G2, mr=0):
    flux = flux0 * 10**(-(mag - mr)/2.5)
    return flux


def get_flux_from_flux_inst(flux_inst, flux0, mr):
    flux = get_flux_from_mag(-2.5*np.log10(flux_inst), flux0=flux0, mr=mr)
    return flux


# flux_inst_ = 104238.6
# mag_cat = 13.145307101711005
# flux_abs_form_mag = get_flux_from_mag(mag_cat, flux0_G2, 0)
# print(f'{flux_abs_form_mag=:.3f}')
#
#
# mag_inst = zero_point_G2 - 2.5*np.log10(flux_inst_)
# print(f'{mag_inst=:.3f}')
#
#
# flux_abs_form_flux_inst = get_flux_from_mag(mag_inst, flux0_G2, 0)
# print(f'{flux_abs_form_flux_inst=:.3f}')
#
#
# flux_abs_form_flux_inst_1 = get_flux_from_flux_inst(flux_inst_, flux0=flux0_G2, mr=-1*zero_point_G2)
# print(f'{flux_abs_form_flux_inst_1=:.3f}')
#
# pass


def extract_timesys(votable_tree):
    """
    Extracts TimeSys attributes (timeorigin, timescale, refposition)
    from the GAVO tree.
    """
    timesys_data = {'timeorigin': 0.0, 'timescale': 'UTC', 'refposition': 'UNKNOWN'}

    def find_timesys(node, text, attrs, childIter):
        if node.name_ == 'TIMESYS':
            # Extraction from attrs dictionary is safest in GAVO
            timesys_data['timeorigin'] = float(attrs.get('timeorigin', 0.0))
            timesys_data['timescale'] = attrs.get('timescale', 'UTC')
            timesys_data['refposition'] = attrs.get('refposition', 'UNKNOWN')

        for child in childIter:
            if hasattr(child, 'apply'):
                child.apply(find_timesys)

    votable_tree.apply(find_timesys)
    return timesys_data


def extract_photcal(votable_tree):
    """
    Extracts multiple PhotCal structures, resolving PARAMrefs by
    performing a pre-scan of all IDs in the VOTable.
    """

    # 1. Pre-scan the tree to map IDs to actual PARAM objects
    id_map = {}

    def map_ids(node, text, attrs, childIter):
        # Every element might have an ID
        node_id = getattr(node, "id", None) or getattr(node, "ID", None)
        if node_id:
            id_map[node_id] = node

        for child in childIter:
            if hasattr(child, 'apply'):
                child.apply(map_ids)

    votable_tree.apply(map_ids)

    # 2. Main Traversal to extract grouped calibrations
    calibrations = {}
    # utypes:
    UT_FLUX = "photDM:PhotCal.zeroPoint.flux.value"
    UT_MAG = "photDM:PhotCal.zeroPoint.referenceMagnitude.value"
    UT_FILTER = "photDM:PhotometryFilter.identifier"

    def process_group(node, text, attrs, childIter):
        if node.name_ == 'GROUP' and getattr(node, "name", None) == "photcal":
            group_data = {'zp_flux': None, 'zp_mag': 0.0, 'filter': ''}
            target_col = None

            for child in childIter:
                # Handle Direct PARAM
                if child.name_ == 'PARAM':
                    ut = getattr(child, "utype", "").lower()
                    if ut == UT_FLUX.lower():
                        group_data['zp_flux'] = float(child.value)
                    elif ut == UT_MAG.lower():
                        group_data['zp_mag'] = float(child.value)
                    elif ut == UT_FILTER.lower():
                        group_data['filter_id'] = child.value

                # Handle PARAMref (Pointing to our id_map)
                elif child.name_ == 'PARAMref':
                    ref_id = getattr(child, "ref", None)
                    referenced_param = id_map.get(ref_id)
                    if referenced_param is not None:
                        ut = getattr(referenced_param, "utype", "").lower()
                        if ut == UT_FLUX.lower():
                            group_data['zp_flux'] = float(referenced_param.value)
                        elif ut == UT_MAG.lower():
                            group_data['zp_mag'] = float(referenced_param.value)
                        elif ut == UT_FILTER.lower():
                            group_data['filter_id'] = referenced_param.value

                # Handle FIELDref
                elif child.name_ == 'FIELDref':
                    target_col = getattr(child, "ref", None)

            if target_col:
                calibrations[target_col] = group_data

        # Recurse
        for child in childIter:
            if hasattr(child, 'apply'):
                child.apply(process_group)

    votable_tree.apply(process_group)
    return calibrations


def get_empty_metadata():
    """Returns the standard structure with default values."""
    return {
        'photcal': {},      # Will stay empty or be filled by heuristics
        'timesys': {
            'timeorigin': 0.0,
            'timescale': 'UTC',
            'refposition': 'HELIOCENTER'
        },
        'jd0': 0.0
    }


def ingest_lightcurve(file_path):
    """
    Unified ingestor
    """

    meta = get_empty_metadata()
    table = None

    try:
        if is_votable(file_path):
            # CASE 1: Good VOTable
            table = Table.read(file_path, format='votable')

            with open(file_path, "rb") as f:
                votable_tree = votparse.readRaw(f)

            # Extract DM metadata
            meta['photcal'] = extract_photcal(votable_tree)
            meta['timesys'] = extract_timesys(votable_tree)
            meta['jd0'] = meta['timesys'].get('timeorigin', 0.0)

        else:
            # CASE 2: Still relying on the Table
            table = Table.read(file_path)
            # Try to pick up JD0 from comments
            meta['jd0'] = pickup_jd0_from_table(table)

    except Exception as e:
        # CASE 3: ASCII fallback
        print(f"Astropy Table failed")
        table = ascii.read(file_path)
        meta['jd0'] = pickup_jd0_from_table(table)

    # 3. Keep metadata
    if table is not None:
        meta['timesys']['timeorigin'] = meta['jd0']
        table = standardize_columns(table)

        # Apply units and UCDs if missing
        table = promote_to_vo_standards(table)

        # Ensure photcal has entries for the standardized columns
        for col in ['mag', 'flux']:
            if col in table.colnames and col not in meta['photcal']:
                meta['photcal'][col] = {'zp_flux': None, 'zp_mag': 0.0, 'filter_id': ''}

        table.meta.update(meta)

        for colname in table.colnames:
            print(
                f"Col: {colname:10} Unit: {str(table[colname].unit):10} UCD: {table[colname].info.meta.get('ucd', 'None')}")

    return table


def pickup_jd0_from_table(table):
    """Scans the table's existing comment metadata for JD0."""
    jd0_pattern = re.compile(r"JD0\s*=\s*([+-]?\d*\.?\d+)")

    # Astropy puts header '#' lines here
    comments = table.meta.get('comments', [])

    for line in comments:
        match = jd0_pattern.search(line.upper())
        if match:
            return float(match.group(1))
    return 0.0


def standardize_columns(table):
    """Maps aliases to standard internal column names."""
    column_map = {
        'obs_time': ['time', 'jd', 'mjd', 'col1'],
        'mag': ['mag', 'magnitude', 'm', 'col2'],
        'flux': ['flux', 'f', 'counts'],
        'err': ['err', 'error', 'magerr', 'mag_err', 'mag_error', 'fluxerr', 'flux_err', 'flux_error', 'col3']
    }
    for standard_name, aliases in column_map.items():
        if standard_name in table.colnames:
            continue
        for alias in aliases:
            if alias in table.colnames:
                table.rename_column(alias, standard_name)
                break
    return table


def get_magnitude_column(table):
    # Search the table for the column tagged as phot.mag
    for colname in table.colnames:
        if table[colname].info.meta and 'phot.mag' in table[colname].info.meta.get('ucd'):
            return table[colname]
    return None


def get_magnitude_error_column(table):
    # Search the table for the column tagged as phot.mag
    for colname in table.colnames:
        if ((table[colname].info.meta and
                'phot.mag' in table[colname].info.meta.get('ucd')) and
                'stat.error' in table[colname].info.meta.get('ucd')):
            return table[colname]
    return None


def get_time_column(table):
    # Search the table for the column tagged as phot.mag
    for colname in table.colnames:
        if table[colname].info.meta and 'time.epoch' in table[colname].info.meta.get('ucd'):
            return table[colname]
    return None


def promote_to_vo_standards(table):
    """
    Assigns units and UCDs to columns ONLY if they are missing.
    """
    for colname in table.colnames:
        col = table[colname]
        name_lower = colname.lower()

        # Check current state
        has_unit = col.unit is not None
        has_ucd = (col.info.meta is not None) and (col.info.meta.get('ucd') is not None)

        # 1. Determine the base concept (only if UCD is missing)
        base_ucd = None
        suggested_unit = None

        if 'mag' in name_lower:
            base_ucd = 'phot.mag'
            suggested_unit = u.mag
        elif 'flux' in name_lower:
            base_ucd = 'phot.flux'
            suggested_unit = u.s**-1
        elif any(k in name_lower for k in ['time', 'jd', 'mjd']):
            base_ucd = 'time.epoch'
            suggested_unit = u.d

        # 2. Apply Heuristics Respectfully
        if base_ucd:
            # ONLY assign UCD if it's missing
            if not has_ucd:
                if col.info.meta is None:
                    col.info.meta = {}

                # Check for error indicators
                if any(k in name_lower for k in ['err', 'uncert', 'sigma']):
                    col.info.meta['ucd'] = f"stat.error;{base_ucd}"
                else:
                    col.info.meta['ucd'] = base_ucd

            # ONLY assign unit if it's missing
            if not has_unit:
                col.unit = suggested_unit

    return table


# from astropy.table import Column
#
#
# def add_abs_flux(table):
#     """
#     Identifies magnitude columns, looks up their PhotCal constants,
#     and adds a new column for flux density (Jy).
#     """
#
#     phot_meta = table.meta.get('photcal', {})
#
#     if not phot_meta:
#         print("Warning: No photcal metadata found. Cannot calculate absolute flux.")
#         return table
#
#     # 2. Iterate through all columns in the table
#     # We use a list of names because we'll be adding columns during iteration
#     for colname in list(table.colnames):
#         col = table[colname]
#
#         # 3. Check if this column has calibration metadata
#         # And ensure it's actually a magnitude (using UCD or heuristics)
#         ucd = col.info.meta.get('ucd', '') if col.info.meta else ''
#
#         if colname in phot_meta:
#             cal = phot_meta[colname]
#
#             # Check if we have the required constants
#             f0 = cal.get('zp_flux')
#             mr = cal.get('zp_mag', 0.0)
#
#             if f0 is not None:
#                 # 4. Perform the calculation
#                 # Using your formula: flux = flux0 * 10**(-(mag - mr)/2.5)
#                 mags = col.data
#                 flux_vals = f0 * 10**(-(mags - mr) / 2.5)
#
#                 # 5. Create the new column name (e.g., mag_flux)
#                 new_colname = f"{colname}_flux_jy"
#
#                 # 6. Add to table with correct units and UCD
#                 new_col = Column(
#                     data=flux_vals,
#                     name=new_colname,
#                     unit='Jy',
#                     description=f"Absolute flux density derived from {colname}"
#                 )
#
#                 # Set metadata for the new column
#                 new_col.info.meta = {'ucd': 'phot.flux.density;stat.filled'}
#
#                 table.add_column(new_col)
#                 print(f"Added absolute flux column: {new_colname}")
#
#     return table

# todo: think -- when we use photcal we must keep original column names

def main():
    for filename in ['data/my_g3.vot', 'data/AY_Lac-R.vot']:
        with open(filename, "rb") as f:
            votable_tree = votparse.readRaw(f)
        res = extract_photcal(votable_tree)
        print(res)

    lc = ingest_lightcurve('data/lc_tess_HD182144_TIC_406949643_sector__40_author__SPOC_methods__pdcsap.vot')
    print(lc.meta)

    lc = ingest_lightcurve('data/OGLE-SMC-CEP-0325-I.vot')
    print(lc.meta)

    # print(lc)
    lc = ingest_lightcurve('data/6009363278148078848-G.vot')
    print(lc.meta)

    # print(lc)
    lc = ingest_lightcurve('data/ASAS19pm/ASas19pm.dat')
    print(lc.meta)


if __name__ == "__main__":
    main()