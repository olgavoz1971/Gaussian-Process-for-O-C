# import volightcurve as vo
import io
import json
import logging
import numpy as np
import pandas as pd
from astropy import units as u
from volightcurve import (VOLightCurve, get_time_colnames, get_flux_colnames, get_mag_colnames,
                          get_error_colnames, is_mag_column, is_flux_column)

logger = logging.getLogger(__name__)

"""
Basic lightcurve data + metadate structure is
{
    "schema": {
        "time": "jd",
        "value": "mag", 
        "error": "mag_err"
    },
    "data": [
        [2459000.5, 15.2, 0.02, "source_1"],
        [2459001.5, 15.3, 0.05, "phase_2"]
        [2459001.5, 15.3, 0.05, "sector 12345"]
    ],
    "meta": {
        "active_domain": "mag",
        "jd0": 0.0,
        "photcal": {
            "zp_flux": 3631.0,
            "zp_flux_unit": "Jy",
            "zp_mag": 0.0,
            "mag_sys": "Vega"
        }
    }
}
"""

# Shared JS Logic for Dash Client-side Callbacks
# todo: this is a stub
JS_CODE = """
window.photUtils = {
    magToFlux: (mag, zp_mag, zp_flux) => zp_flux * Math.pow(10, -0.4 * (mag - zp_mag)),
    fluxToMag: (flux, zp_mag, zp_flux) => zp_mag - 2.5 * Math.log10(flux / zp_flux)
};
"""


def read_to_volc(file_source):
    """
    1) Reads a lightcurve from a file path OR a file-like object (io.BytesIO).
    Returns a VOLightCurve instance.
    """
    try:
        # VOLightCurve internally uses Table.read, which handles file-like objects
        volc = VOLightCurve(file_source)
        return volc
    except Exception as e:
        logger.error(f"VOLightCurve read failed: {e}")
        raise


# We need this to tame our float types zoo
class LCEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles NumPy types (float32, int64, etc.)
    and Astropy Quantities.
    """

    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Fallback to the standard encoder
        return super().default(obj)


def pack_volc_to_json(lc: VOLightCurve, primary_col=None, error_col=None):
    """
    Packs a VOLightCurve into an economic JSON schema for Web/JS transport.
    """

    mag_cols = get_mag_colnames(lc)
    flux_cols = get_flux_colnames(lc)
    time_cols = get_time_colnames(lc)
    error_cols = get_error_colnames(lc)

    if not time_cols:
        raise ValueError("No time columns found in VOLightCurve.")

    if not primary_col:
        primary_col = mag_cols[0] if mag_cols else (flux_cols[0] if flux_cols else None)

    if not primary_col:
        raise ValueError("No magnitude or flux columns found in VOLightCurve.")

    # todo: this is a stub! Use a proper error column (mag or flux or calculate mag_err from flux_err (Gaia case)
    if is_mag_column(lc.table, primary_col):
        error_cols = lc.get_mag_error_colnames()
    else:
        error_cols = lc.get_flux_error_colnames()

    if not error_col:
        error_col = error_cols[0] if error_cols else None

    # "The Rest" - Flag Column Identification
    meaningful = set(time_cols + mag_cols + flux_cols + error_cols)
    flag_col = next((c for c in lc.table.colnames if c not in meaningful), None)

    # Photometry Calibration Extraction
    photdm = lc.photdms.get(primary_col)
    photcal = photdm.photcal if photdm else None

    photcal_meta = {
        "zp_flux": photcal.zp_flux.value if photcal else 1.0,
        "zp_flux_unit": photcal.zp_flux.unit.to_string('vounit') if photcal else '',
        "zp_mag": photcal.zp_mag.value if photcal else 0.0,
        "zp_mag_unit": photcal.zp_mag.unit.to_string('vounit') if photcal else '',
        "mag_sys": photcal.mag_sys if photcal else 'Unknown'
    }

    # Data Extraction
    # .value is used to strip Astropy units before JSON serialization
    time_col = time_cols[0]
    t = lc[time_col].value
    v = lc[primary_col].value

    # Use .tolist() or raw arrays to handle potential None/nulls safely
    e = lc[error_col].value if error_col else [None] * len(v)

    # Flags can be mixed types; .value (if it's a Quantity) or raw column
    f_data = lc.table[flag_col] if flag_col else [None] * len(v)
    f = f_data.value if hasattr(f_data, 'value') else f_data

    # Final Construction
    struct = {
        "schema": {
            "time": time_col,
            "value": primary_col,
            "error": error_col,  # None if missing
            "flag": flag_col,  # None if missing
        },
        "data": [list(row) for row in zip(t, v, e, f)],
        "meta": {
            "active_domain": "mag" if primary_col in mag_cols else "flux",
            "jd0": lc.timesys.jd0,
            "photcal": photcal_meta
        }
    }

    return json.dumps(struct, cls=LCEncoder)


def unpack_json_for_plotly(json_str: str, view_mode='mag'):
    """
    Lean NumPy version: Decodes JSON and prepares data for Plotly
    without the Pandas overhead.
    """
    packet = json.loads(json_str)
    meta = packet['meta']
    photcal = meta['photcal']

    # Convert list of lists to a 2D NumPy array
    # Array indices: 0=time, 1=value, 2=error, 3=flag
    data = np.array(packet['data'], dtype=object)

    t = data[:, 0].astype(float)
    v = data[:, 1].astype(float)
    # Handle the 'None' in errors safely
    e_raw = data[:, 2]
    has_err = packet['schema']['error'] is not None
    e = e_raw.astype(float) if has_err else None

    f = data[:, 3] # Keep flags as objects/strings

    # Apply JD0
    # jd0 = meta.get('jd0', 0)
    # if jd0:
    #     t += jd0

    # 2. Domain Logic
    current_domain = meta['active_domain']
    y_data = v
    e_data = e

    if view_mode != current_domain:
        # zp_m = photcal['zp_mag']
        # todo: !!!!!!!!!
        zp_m = 25.0
        zp_f = photcal['zp_flux']

        if view_mode == 'flux' and current_domain == 'mag':
            # Mag -> Flux
            y_data = zp_f * 10**(-0.4 * (v - zp_m))
            if has_err:
                # sigma_f = f * 0.921 * sigma_m
                e_data = y_data * 0.921034 * e

        elif view_mode == 'mag' and current_domain == 'flux':
            # Flux -> Mag (Safeguard against zero/negative flux)
            mask = v > 0
            y_data = np.full_like(v, np.nan)
            y_data[mask] = zp_m - 2.5 * np.log10(v[mask] / zp_f)

            if has_err:
                e_data = np.full_like(e, np.nan)
                e_data[mask] = 1.085736 * (e[mask] / v[mask])

    return {
        'x': t,
        'y': y_data,
        'err': e_data,
        'flag': f,
        'x_label': "Julian Date (JD)",
        'y_label': "Magnitude" if view_mode == 'mag' else "Flux",
        'is_mag': (view_mode == 'mag')
    }


def get_flux_fragment(json_str: str, jd_min: float, jd_max: float) -> pd.DataFrame:
    """
    Extracts a JD-sliced fragment and ensures it is in FLUX domain
    for the GP module. Returns a pandas DataFrame.
    """
    # We force view_mode='flux' so the bridge handles the math
    lc = unpack_json_for_plotly(json_str, view_mode='flux')

    # 2. Build the DataFrame
    df = pd.DataFrame({
        'jd': lc['x'],
        'flux': lc['y'],
        'flux_err': lc['err'] if lc['err'] is not None else np.nan
    })

    # Slice by JD
    mask = (df['jd'] >= jd_min) & (df['jd'] <= jd_max)
    frag = df[mask].copy()

    return frag


# def unpack_json_for_plotly(json_str: str, view_mode='mag'):
#     """
#     Decodes the transport JSON and prepares data for Plotly.
#     Handles the Mag <-> Flux conversion if the requested view_mode
#     differs from the active_domain in the JSON.
#     """
#     packet = json.loads(json_str)
#     schema = packet['schema']
#     meta = packet['meta']
#     photcal = meta['photcal']
#
#     # 1. Load into DataFrame
#     # Columns are: [time, value, error, flag]
#     df = pd.DataFrame(packet['data'], columns=['t', 'v', 'e', 'f'])
#
#     # 2. Re-apply JD0 if present
#     jd0 = meta.get('jd0', 0)
#     if jd0:
#         df['t'] += jd0
#
#     # 3. Domain Logic: Do we need to convert?
#     # current_domain is 'mag' or 'flux'
#     current_domain = meta['active_domain']
#
#     y_data = df['v']
#     e_data = df['e']
#
#     if view_mode != current_domain:
#         # We need to calculate the other domain
#         zp_m = photcal['zp_mag']
#         zp_f = photcal['zp_flux']
#
#         if view_mode == 'flux' and current_domain == 'mag':
#             # Mag -> Flux: F = F0 * 10^(-0.4 * (m - m0))
#             y_data = zp_f * 10 ** (-0.4 * (df['v'] - zp_m))
#             # Error propagation (approximate): sigma_f = f * 0.921 * sigma_m
#             if schema['error']:
#                 e_data = y_data * 0.921034 * df['e']
#
#         elif view_mode == 'mag' and current_domain == 'flux':
#             # Flux -> Mag: m = m0 - 2.5 * log10(F / F0)
#             y_data = zp_m - 2.5 * np.log10(df['v'] / zp_f)
#             # Error propagation: sigma_m = 1.0857 * (sigma_f / f)
#             if schema['error']:
#                 e_data = 1.085736 * (df['e'] / df['v'])
#
#     return {
#         'x': df['t'],
#         'y': y_data,
#         'err': e_data if schema['error'] else None,
#         'flag': df['f'],
#         'x_label': "Julian Date (JD)",
#         'y_label': "Magnitude" if view_mode == 'mag' else "Flux",
#         'is_mag': (view_mode == 'mag')
#     }


def pretty_print_lc_json(json_str: str, max_rows: int = 5):
    """
    Parses the LC JSON string and prints a human-readable summary.
    Summarizes the 'data' array and handles missing columns/None values cleanly.
    """
    try:
        packet = json.loads(json_str)
    except json.JSONDecodeError:
        print("Error: Input is not a valid JSON string.")
        return

    schema = packet.get("schema", {})
    meta = packet.get("meta", {})
    photcal = meta.get("photcal", {})
    data = packet.get("data", [])

    print("=" * 80)
    print(f"{'VOLightCurve JSON Transport Package':^80}")
    print("=" * 80)

    # 1. Metadata Section
    print(f"PRIMARY DOMAIN: {meta.get('active_domain', 'Unknown').upper()}")
    print(f"JD0 (Offset):   {meta.get('jd0', 'N/A')}")
    print(f"CALIBRATION:    Sys: {photcal.get('mag_sys', 'N/A')}")
    print(f"                ZP Mag:  {photcal.get('zp_mag')} {photcal.get('zp_mag_unit')}")
    print(f"                ZP Flux: {photcal.get('zp_flux')} {photcal.get('zp_flux_unit')}")
    print("-" * 80)

    # 2. Schema Section
    print("SCHEMA / COLUMN MAPPING:")
    for key, colname in schema.items():
        # Display "[Not Provided]" if the colname is None/Null
        label = colname if colname else "[Not Provided]"
        print(f"  {key:10} -> {label}")
    print("-" * 80)

    # 3. Data Summary Section
    total_rows = len(data)
    print(f"DATA ({total_rows} rows):")

    # Clean Headers: Use the actual column name or a placeholder if missing
    h_time = schema.get('time') or "Time"
    h_val = schema.get('value') or "Value"
    h_err = schema.get('error') or "Error"
    h_flag = schema.get('flag') or "Flag"

    headers = [h_time, h_val, h_err, h_flag]
    # Increased width to 18 to handle long JDs and names
    print(f"  {' | '.join([f'{h:^18}' for h in headers])}")
    print(f"  {'-' * 78}")

    def format_row(row):
        parts = []
        for item in row:
            if item is None:
                parts.append(f"{'-':^18}")
            elif isinstance(item, float):
                # 6 decimal places is plenty for a quick look
                parts.append(f"{item:^18.6f}")
            else:
                parts.append(f"{str(item):^18}")
        return " | ".join(parts)

    # Rows (head and tail)
    if total_rows > (max_rows * 2):
        for row in data[:max_rows]:
            print(f"  {format_row(row)}")

        divider = [". . ."] * 4
        print(f"  {' | '.join([f'{d:^18}' for d in divider])}")

        for row in data[-max_rows:]:
            print(f"  {format_row(row)}")
    else:
        for row in data:
            print(f"  {format_row(row)}")

    print("=" * 80)


# def pretty_print_lc_json(json_str: str, max_rows: int = 5):
#     """
#     Parses the LC JSON string and prints a human-readable summary.
#     Summarizes the 'data' array to avoid flooding the console.
#     """
#     try:
#         packet = json.loads(json_str)
#     except json.JSONDecodeError:
#         print("Error: Input is not a valid JSON string.")
#         return
#
#     schema = packet.get("schema", {})
#     meta = packet.get("meta", {})
#     photcal = meta.get("photcal", {})
#     data = packet.get("data", [])
#
#     print("=" * 60)
#     print(f"{'VOLightCurve JSON Transport Package':^60}")
#     print("=" * 60)
#
#     # 1. Metadata Section
#     print(f"PRIMARY DOMAIN: {meta.get('active_domain', 'Unknown').upper()}")
#     print(f"JD0 (Offset):   {meta.get('jd0', 'N/A')}")
#     print(f"CALIBRATION:    Sys: {photcal.get('mag_sys', 'N/A')}")
#     print(f"                ZP Mag:  {photcal.get('zp_mag')} {photcal.get('zp_mag_unit')}")
#     print(f"                ZP Flux: {photcal.get('zp_flux')} {photcal.get('zp_flux_unit')}")
#     print("-" * 60)
#
#     # 2. Schema Section
#     print("SCHEMA / COLUMN MAPPING:")
#     for key, colname in schema.items():
#         print(f"  {key:10} -> {colname}")
#     print("-" * 60)
#
#     # 3. Data Summary Section
#     total_rows = len(data)
#     print(f"DATA ({total_rows} rows):")
#
#     # Headers
#     headers = [schema.get('time', 'time'), schema.get('value', 'val'),
#                schema.get('error', 'err'), schema.get('flag', 'flag')]
#     print(f"  {' | '.join([f'{h:^12}' for h in headers])}")
#     print(f"  {'-' * 55}")
#
#     # Rows (head and tail)
#     if total_rows > (max_rows * 2):
#         display_rows = data[:max_rows]
#         for row in display_rows:
#             print(f"  {' | '.join([f'{str(item):^12}' for item in row])}")
#         print(f"  {'...':^12} | {'...':^12} | {'...':^12} | {'...':^12}")
#         display_rows = data[-max_rows:]
#         for row in display_rows:
#             print(f"  {' | '.join([f'{str(item):^12}' for item in row])}")
#     else:
#         for row in data:
#             print(f"  {' | '.join([f'{str(item):^12}' for item in row])}")
#
#     print("=" * 60)


def main():
    # filename = 'data/ASAS19pm/ASas19pm.DAT'
    # lc = vo.VOLightCurve(file_path=filename)
    # print(lc)
    for filename in [
        # 'data/lc_tess_HD182144_TIC_406949643_sector__40_author__SPOC_methods__pdcsap.vot',
        # 'data/OGLE-SMC-CEP-0325-I.vot',
        # 'data/6009363278148078848-G.vot',
        # 'data/AY_Lac-R.vot',
        # 'data/g2_jk.vot',
        # 'data/my_g3.vot',
        'data/ASAS19pm/ASas19pm.dat'
    ]:
        print(f'\n\n\n Ingesting {filename}')

        lc1 = read_to_volc(filename)
        print(lc1)
        json_str = pack_volc_to_json(lc1)
        pretty_print_lc_json(json_str)


if __name__ == "__main__":
    main()
