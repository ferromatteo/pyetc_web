from flask import Flask, render_template, request, jsonify
from pyetc_wst.wst import WST
import warnings
import traceback
import json
import numpy as np
warnings.filterwarnings('ignore')

app = Flask(__name__)

# All possible instruments and channels
INSTRUMENTS = ['ifs', 'moshr', 'moslr']
CHANNELS = {
    'ifs': ['blue', 'red'],
    'moslr': ['blue', 'green', 'red'],
    'moshr': ['U', 'B', 'V', 'I']
}

# Color mapping for plots (matching HTML colors)
COLORS = {
    'ifs-red': '#c62828',
    'ifs-blue': '#1565c0',
    'moshr-U': '#6a1b9a',
    'moshr-B': '#01579b',
    'moshr-V': '#2e7d32',
    'moshr-I': '#d84315',
    'moslr-blue': '#0d47a1',
    'moslr-green': '#388e3c',  
    'moslr-red': '#b71c1c'
}

# All parameter keys (excluding INS/CHAN)
ALL_PARAM_KEYS = [
    "NDIT", "DIT", "SNR", "Lam_Ref", "OBJ_FIB_DISP", "MOON", "PWV", "FLI", "SEE", "AM", "SKYCALC",
    "Obj_SED", "SED_Name", "OBJ_MAG", "MAG_SYS", "MAG_FIL", "Z", "BB_Temp", "PL_Index", 
    "SEL_FLUX", "SEL_CWAV", "SEL_FWHM",
    "Obj_Spat_Dis", "IMA", "Ext_Ell", "IMA_FWHM", "IMA_BETA", "IMA_KFWHM", "Sersic_Reff", "Sersic_Ind", "IMA_KREFF",
    "SPEC_RANGE", "SPEC_KFWHM", "SPEC_HSIZE", "COADD_WL", "IMA_RANGE", "COADD_XY", "OPT_SPEC", "OPT_IMA", "FRAC_SPEC_MEAN_OPT_IMAGE"
]

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    res_time = None
    res_snr = None
    debug_output = ""
    plot_data = None
    
    # Default values for all parameters
    default_params = {
        "NDIT": 1,
        "DIT": 600, 
        "SNR": 10,
        "Lam_Ref": 5000,
        "OBJ_FIB_DISP": 0,
        "MOON": None,
        "PWV": 10,
        "FLI": 0,
        "SEE": 0.8,
        "AM": 1.,
        "SKYCALC": True,
        "Obj_SED": 'template',
        "SED_Name": 'Kinney_s0',
        "OBJ_MAG": 12,
        "MAG_SYS": 'Vega',
        "MAG_FIL": 'V',
        "Z": 0,
        "BB_Temp": 9000.,
        "PL_Index": -2,
        "SEL_FLUX": 50e-16,
        "SEL_CWAV": 5000,
        "SEL_FWHM": 20,
        "Obj_Spat_Dis": 'ps',
        "IMA": 'sersic',
        "Ext_Ell": None,
        "IMA_FWHM": None,
        "IMA_BETA": None,
        "IMA_KFWHM": None,
        "Sersic_Reff": 3.0,
        "Sersic_Ind": 1.0,
        "IMA_KREFF": 5,
        "SPEC_RANGE": 'fixed',
        "SPEC_KFWHM": None,
        "SPEC_HSIZE": 999999,
        "COADD_WL": 1,
        "IMA_RANGE": 'square_fixed',
        "COADD_XY": 1,
        "OPT_SPEC": False,
        "OPT_IMA": False,
        "FRAC_SPEC_MEAN_OPT_IMAGE": 1
    }
    
    params = default_params.copy()
    configs = []
    
    if request.method == 'POST':
        try:
            # Get selected configurations
            selected_configs = request.form.getlist('config')
            
            # Get compute mode
            compute_mode = request.form.get('compute_mode', 'dit_ndit')

            # Persist compute_mode in params so it stays selected after Compute
            params['compute_mode'] = compute_mode
            
            # If no configuration is selected, show warning
            if not selected_configs:
                debug_output = "ERROR: No configuration selected. Please select at least one instrument-channel pair."
                return render_template('index.html', result=result, res_time=res_time, res_snr=res_snr, params=params, debug_output=debug_output, plot_data=plot_data)
            
            # Parse selected instrument-channel pairs and create configs
            configs = []
            for pair in selected_configs:
                inst, chan = pair.split('-')
                config = default_params.copy()
                config['INS'] = inst
                config['CH'] = chan
                configs.append(config)
            
            # Update all configs and params with user-provided values
            for config in configs:
                for k in ALL_PARAM_KEYS:
                    v = request.form.get(k)
                    if v is not None and v != '':
                        if v == 'True':
                            config[k] = True
                            params[k] = True
                        elif v == 'False':
                            config[k] = False
                            params[k] = False
                        elif v.replace('.', '', 1).replace('-', '', 1).replace('e', '', 1).replace('E', '', 1).replace('+','',1).isdigit():
                            try:
                                # Check if it's a float or int
                                if '.' in v or 'e' in v.lower():
                                    config[k] = float(v)
                                    params[k] = float(v)
                                else:
                                    config[k] = int(v)
                                    params[k] = int(v)
                            except:
                                config[k] = v
                                params[k] = v
                        else:
                            config[k] = v
                            params[k] = v
            
            # Run ETC for each configuration
            debug_lines = []
            debug_lines.append("=" * 80)
            debug_lines.append("WST ETC - COMPUTATION RESULTS")
            debug_lines.append("=" * 80)
            debug_lines.append("")
            
            # Create a single WST object
            obj = WST(log='DEBUG', skip_dataload=False)
            
            # Store plot data
            plot_traces = []
            summary_table = []
            has_errors = False
            
            for idx, config in enumerate(configs):
                try:
                    inst = config['INS']
                    chan = config['CH']
                    config_key = f"{inst}-{chan}"
                    
                    debug_lines.append(f"Configuration {idx+1}: {inst.upper()} - {chan.upper()}")
                    debug_lines.append("-" * 80)
                    # Assign is_ifs and is_mos before using them
                    is_ifs = inst.lower() == 'ifs'
                    is_mos = inst.lower() in ['moshr', 'moslr']
                    # Show number of spaxels for IFS
                    if is_ifs:
                        n_spaxels = config.get('COADD_XY', params.get('COADD_XY', 1))
                        debug_lines.append(f"  Number of spaxels (spatial coadding): {n_spaxels}x{n_spaxels}")
                    
                    # Build observation
                    con, ob, spe, im, spe_input = obj.build_obs_full(config)
                    
                    # Determine if IFS or MOS
                    is_ifs = inst.lower() == 'ifs'
                    is_mos = inst.lower() in ['moshr', 'moslr']
                    
                    # Store results
                    res_result = None
                    computed_snr = None
                    computed_time = None
                    
                    # Compute based on mode
                    coadd_wl = config.get('COADD_WL', params.get('COADD_WL', 1))
                    
                    if compute_mode == 'dit_ndit':
                        # Compute SNR from DIT & NDIT
                        debug_lines.append(f"  Mode: DIT & NDIT")
                        debug_lines.append(f"  DIT: {config['DIT']} s")
                        debug_lines.append(f"  NDIT: {config['NDIT']}")
                        
                        if is_ifs:
                            res_result = obj.snr_from_source(con, im, spe)
                        elif is_mos:
                            res_result = obj.snr_from_source_MOS(con, im, spe)
                        
                        # Check if result contains error message
                        if 'message' in res_result:
                            debug_lines.append(f"  ⚠ WARNING: {res_result['message']}")
                            if 'frac_sat' in res_result:
                                    if is_ifs:
                                        debug_lines.append(f"  → Fraction of saturated voxels: {res_result['frac_sat']*100:.1f}%")
                                    elif is_mos:
                                        debug_lines.append(f"  → Fraction of saturated pixels: {res_result['frac_sat']*100:.1f}%")
                            has_errors = True
                        else:
                            computed_snr = res_result
                            wave_array = res_result['spec']['snr'].wave.coord()
                            snr_array = res_result['spec']['snr'].data.data
                            # Use SEL_CWAV as reference wavelength if Obj_SED is 'line', else Lam_Ref
                            if config.get('Obj_SED', params.get('Obj_SED', 'template')) == 'line':
                                ref_wave = config.get('SEL_CWAV', params.get('SEL_CWAV', 7000))
                            else:
                                ref_wave = 0.5 * (wave_array[-1] + wave_array[0])
                            idx_closest = (np.abs(wave_array - ref_wave)).argmin()
                            true_wave = wave_array[idx_closest]
                            achieved_snr = snr_array[idx_closest]
                            # Always print SNR per pixel
                            debug_lines.append(f"  → Achieved SNR at central wavelength {true_wave:.1f} Å: {achieved_snr:.2f}")
                            # Print SNR with spectral coadding if available
                            if 'snr_rebin' in res_result['spec']:
                                snr_array_rebin = res_result['spec']['snr_rebin'].data.data
                                debug_lines.append(f"  → Achieved SNR at central wavelength {true_wave:.1f} Å (with spectral coadding): {snr_array_rebin[idx_closest]:.2f}")
                            # Add saturation info
                            if 'frac_sat' in res_result:
                                    if is_ifs:
                                        debug_lines.append(f"  → Fraction of saturated voxels: {res_result['frac_sat']*100:.1f}%")
                                    elif is_mos:
                                        debug_lines.append(f"  → Fraction of saturated pixels: {res_result['frac_sat']*100:.1f}%")
                            
                    elif compute_mode == 'dit_snr':
                        # Compute NDIT from DIT & SNR
                        debug_lines.append(f"  Mode: DIT & SNR")
                        debug_lines.append(f"  DIT: {config['DIT']} s")
                        debug_lines.append(f"  Target SNR: {config['SNR']}")
                        
                        if is_ifs:
                            computed_time = obj.time_from_source(con, im, spe, dit=False)
                        elif is_mos:
                            computed_time = obj.time_from_source_MOS(con, im, spe, dit=False)
                        
                        # Check if result contains error message
                        if 'message' in computed_time:
                            debug_lines.append(f"  ⚠ WARNING: {computed_time['message']}")
                            if 'frac_sat' in computed_time:
                                        if is_ifs:
                                            debug_lines.append(f"  → Fraction of saturated voxels: {computed_time['frac_sat']*100:.1f}%")
                                        elif is_mos:
                                            debug_lines.append(f"  → Fraction of saturated pixels: {computed_time['frac_sat']*100:.1f}%")
                            has_errors = True
                        else:
                            debug_lines.append(f"  → Required NDIT: {computed_time['ndit']:.2f}")
                            # Add saturation info
                            if 'frac_sat' in computed_time:
                                        if is_ifs:
                                            debug_lines.append(f"  → Fraction of saturated voxels: {computed_time['frac_sat']*100:.1f}%")
                                        elif is_mos:
                                            debug_lines.append(f"  → Fraction of saturated pixels: {computed_time['frac_sat']*100:.1f}%")
                            # Update config with computed NDIT
                            config['NDIT'] = int(np.ceil(computed_time['ndit']))
                            con, ob, spe, im, spe_input = obj.build_obs_full(config)
                            # Compute achieved SNR
                            if is_ifs:
                                res_result = obj.snr_from_source(con, im, spe)
                            elif is_mos:
                                res_result = obj.snr_from_source_MOS(con, im, spe)
                            
                            # Check again for error message
                            if 'message' in res_result:
                                debug_lines.append(f"  ⚠ WARNING: {res_result['message']}")
                                if 'frac_sat' in res_result:
                                        if is_ifs:
                                            debug_lines.append(f"  → Fraction of saturated voxels: {res_result['frac_sat']*100:.1f}%")
                                        elif is_mos:
                                            debug_lines.append(f"  → Fraction of saturated pixels: {res_result['frac_sat']*100:.1f}%")
                                has_errors = True
                            else:
                                computed_snr = res_result
                                # Use SEL_CWAV as reference wavelength if Obj_SED is 'line', else Lam_Ref
                                if config.get('Obj_SED', params.get('Obj_SED', 'template')) == 'line':
                                    ref_wave = config.get('SEL_CWAV', params.get('SEL_CWAV', 7000))
                                else:
                                    ref_wave = config.get('Lam_Ref', params.get('Lam_Ref', 7000))
                                wave_array = res_result['spec']['snr'].wave.coord()
                                snr_array = res_result['spec']['snr'].data.data
                                idx_closest = (np.abs(wave_array - ref_wave)).argmin()
                                true_wave = wave_array[idx_closest]
                                achieved_snr = snr_array[idx_closest]
                                # Always print SNR per pixel
                                debug_lines.append(f"  → Achieved SNR at wavelength {true_wave:.1f} Å\n (closest to requested reference wavelength {ref_wave} Å): {achieved_snr:.2f}")
                                # Print SNR with spectral coadding if available
                                if 'snr_rebin' in res_result['spec']:
                                    snr_array_rebin = res_result['spec']['snr_rebin'].data.data
                                    debug_lines.append(f"  → Achieved SNR at wavelength {true_wave:.1f} Å\n (closest to requested reference wavelength {ref_wave} Å, with spectral coadding): {snr_array_rebin[idx_closest]:.2f}")
                    
                    elif compute_mode == 'ndit_snr':
                        # Compute DIT from NDIT & SNR
                        debug_lines.append(f"  Mode: NDIT & SNR")
                        debug_lines.append(f"  NDIT: {config['NDIT']}")
                        debug_lines.append(f"  Target SNR: {config['SNR']}")
                        
                        if is_ifs:
                            computed_time = obj.time_from_source(con, im, spe, dit=True)
                        elif is_mos:
                            computed_time = obj.time_from_source_MOS(con, im, spe, dit=True)
                        
                        # Check if result contains error message
                        if 'message' in computed_time:
                            debug_lines.append(f"  ⚠ WARNING: {computed_time['message']}")
                            if 'frac_sat' in computed_time:
                                if is_ifs:
                                    debug_lines.append(f"  → Fraction of saturated voxels: {computed_time['frac_sat']*100:.1f}%")
                                elif is_mos:
                                    debug_lines.append(f"  → Fraction of saturated pixels: {computed_time['frac_sat']*100:.1f}%")
                            has_errors = True
                        else:
                            debug_lines.append(f"  → Required DIT: {computed_time['dit']:.2f} s")
                            # Add saturation info
                            if 'frac_sat' in computed_time:
                                if is_ifs:
                                    debug_lines.append(f"  → Fraction of saturated voxels: {computed_time['frac_sat']*100:.1f}%")
                                elif is_mos:
                                    debug_lines.append(f"  → Fraction of saturated pixels: {computed_time['frac_sat']*100:.1f}%")
                            # Update config with computed DIT
                            config['DIT'] = computed_time['dit']
                            con, ob, spe, im, spe_input = obj.build_obs_full(config)
                            # Compute achieved SNR
                            if is_ifs:
                                res_result = obj.snr_from_source(con, im, spe)
                            elif is_mos:
                                res_result = obj.snr_from_source_MOS(con, im, spe)
                            
                            # Check again for error message
                            if 'message' in res_result:
                                debug_lines.append(f"  ⚠ WARNING: {res_result['message']}")
                                if 'frac_sat' in res_result:
                                        if is_ifs:
                                            debug_lines.append(f"  → Fraction of saturated voxels: {res_result['frac_sat']*100:.1f}%")
                                        elif is_mos:
                                            debug_lines.append(f"  → Fraction of saturated pixels: {res_result['frac_sat']*100:.1f}%")
                                has_errors = True
                            else:
                                computed_snr = res_result
                                # Use SEL_CWAV as reference wavelength if Obj_SED is 'line', else Lam_Ref
                                if config.get('Obj_SED', params.get('Obj_SED', 'template')) == 'line':
                                    ref_wave = config.get('SEL_CWAV', params.get('SEL_CWAV', 7000))
                                else:
                                    ref_wave = config.get('Lam_Ref', params.get('Lam_Ref', 7000))
                                wave_array = res_result['spec']['snr'].wave.coord()
                                snr_array = res_result['spec']['snr'].data.data
                                idx_closest = (np.abs(wave_array - ref_wave)).argmin()
                                true_wave = wave_array[idx_closest]
                                achieved_snr = snr_array[idx_closest]
                                if 'snr_rebin' in res_result['spec']:
                                    snr_array_rebin = res_result['spec']['snr_rebin'].data.data
                                    debug_lines.append(f"  → Achieved SNR at wavelength {true_wave:.1f} Å\n (closest to requested reference wavelength {ref_wave} Å, with spectral coadding): {snr_array_rebin[idx_closest]:.2f}")                            
                                debug_lines.append(f"  → Achieved SNR at wavelength {true_wave:.1f} Å\n (closest to requested reference wavelength {ref_wave} Å): {achieved_snr:.2f}")
                    
                    # Extract plot data only if no error
                    if computed_snr is not None and 'spec' in computed_snr:
                        wave = computed_snr['spec']['snr'].wave.coord()
                        snr_data = computed_snr['spec']['snr'].data.data
                        wave_list = wave.tolist() if hasattr(wave, 'tolist') else list(wave)
                        snr_list = snr_data.tolist() if hasattr(snr_data, 'tolist') else list(snr_data)
                        plot_traces.append({
                            'x': wave_list,
                            'y': snr_list,
                            'name': f"{inst.upper()} {chan.upper()} (SNR x spectral pixel)",
                            'color': COLORS.get(config_key, '#000000')
                        })
                        coadd_wl = config.get('COADD_WL', params.get('COADD_WL', 1))
                        if coadd_wl > 1 and 'snr_rebin' in computed_snr['spec']:
                            wave_rebin = computed_snr['spec']['snr_rebin'].wave.coord()
                            snr_rebin = computed_snr['spec']['snr_rebin'].data.data
                            wave_rebin_list = wave_rebin.tolist() if hasattr(wave_rebin, 'tolist') else list(wave_rebin)
                            snr_rebin_list = snr_rebin.tolist() if hasattr(snr_rebin, 'tolist') else list(snr_rebin)
                            plot_traces.append({
                                'x': wave_rebin_list,
                                'y': snr_rebin_list,
                                'name': f"{inst.upper()} {chan.upper()} (SNR x spectral coadding [{coadd_wl} pixels])",
                                'color': COLORS.get(config_key, '#000000'),
                                'secondary': True
                            })
                        
                        # Get frac_sat from the appropriate source
                        frac_sat_val = None
                        if 'frac_sat' in computed_snr:
                            frac_sat_val = computed_snr['frac_sat']
                        elif computed_time and 'frac_sat' in computed_time:
                            frac_sat_val = computed_time['frac_sat']
                        elif res_result and 'frac_sat' in res_result:
                            frac_sat_val = res_result['frac_sat']
                        
                        # Add to summary table
                        summary_row = {
                            'config': f"{inst.upper()} {chan.upper()}",
                            'dit': config['DIT'],
                            'ndit': config['NDIT'],
                            'snr_target': config.get('SNR', '-'),
                            'snr_achieved': f"{snr_data[len(snr_data)//2]:.2f}",
                            'frac_sat': f"{frac_sat_val*100:.1f}%" if frac_sat_val is not None else '-'
                        }
                        summary_table.append(summary_row)
                    
                    debug_lines.append("")
                    
                except Exception as e:
                    debug_lines.append(f"  ERROR: {str(e)}")
                    debug_lines.append(f"  Traceback: {traceback.format_exc()}")
                    debug_lines.append("")
                    has_errors = True
            
            debug_lines.append("=" * 80)
            if has_errors:
                debug_lines.append("Computation completed with warnings/errors (see above)")
            else:
                debug_lines.append("Computation completed successfully")
            debug_lines.append("=" * 80)
            
            debug_output = '\n'.join(debug_lines)
            
            # Prepare plot data for frontend only if we have valid traces
            if plot_traces:
                plot_data = {
                    'traces': plot_traces,
                    'summary': summary_table,
                    'compute_mode': compute_mode
                }
            
        except Exception as e:
            debug_output = f"CRITICAL ERROR: {str(e)}\n\nFull traceback:\n{traceback.format_exc()}"
    
    return render_template('index.html', result=result, res_time=res_time, res_snr=res_snr, params=params, debug_output=debug_output, plot_data=plot_data)

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)