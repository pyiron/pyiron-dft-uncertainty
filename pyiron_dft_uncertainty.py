import numpy as np
import pandas
from pymatgen.ext.matproj import MPRester
from ase.data import reference_states, atomic_numbers
from pyiron_base import Settings
from pyiron_atomistics import Project
from pyiron_atomistics.atomistics.master.murnaghan import eV_div_A3_to_GPa
from pyiron_atomistics.vasp.potential import VaspPotential


delta_project_recommendations = """\
H 900 15
He 600 21
Li 600 15
Be 700 21
B 600 13
C 600 17
N 1200 11
O 1200 15
F 1300 13
Ne 500 13
Na 600 15
Mg 600 21
Al 400 13
Si 400 15
P 400 15
S 400 19
Cl 400 13
Ar 400 13
K 400 15
Ca 400 13
Sc 400 21
Ti 400 21
V 500 15
Cr 500 15
Mn 500 13
Fe 500 15
Co 500 21
Ni 500 13
Cu 700 13
Zn 600 21
Ga 600 11
Ge 400 15
As 400 17
Se 400 13
Br 400 13
Kr 400 13
Rb 400 15
Sr 400 15
Y 400 21
Zr 400 21
Nb 400 15
Mo 400 15 
Tc 500 21
Ru 500 21
Rh 500 13
Pd 500 13
Ag 500 13
Cd 500 21
In 400 19
Sn 400 15
Sb 400 17
Te 400 13
I 400 13
Xe 400 13
Cs 400 15
Ba 400 15
Lu 400 21
Hf 400 21
Ta 400 15
W 500 15
Re 500 21
Os 500 21
Ir 500 13
Pt 500 13
Au 500 13
Hg 400 15
Tl 400 21
Pb 400 13
Bi 400 17
Po 400 19
Rn 400 13"""

# https://cms.mpi.univie.ac.at/vasp/vasp/Recommended_PAW_potentials_DFT_calculations_using_vasp_5_2.html
# Element (and appendix)	default cutoff ENMAX (eV)	valency
basic_pot_text = """\
H	250	1
H AE	1000	1
H h	700	1
H s	200	1
He	479	2
Li	140	1
Li sv	499	3
Be	248	2
Be sv	309	4
B	319	3
B h	700	3
B s	269	3
C	400	4
C h	700	4
C s	274	4
N	400	5
N h	700	5
N s	280	5
O	400	6
O h	700	6
O s	283	6
F	400	7
F h	700	7
F s	290	7
Ne	344	8
Na	102	1
Na pv	260	7
Na sv	646	9
Mg	200	2
Mg pv	404	8
Mg sv	495	10
Al	240	3
Si	245	4
P	255	5
P h	390	5
S	259	6
S h	402	6
Cl	262	7
Cl h	409	7
Ar	266	8
K pv	117	7
K sv	259	9
Ca pv	120	8
Ca sv	267	10
Sc	155	3
Sc sv	223	11
Ti	178	4
Ti pv	222	10
Ti sv	275	12
V	193	5
V pv	264	11
V sv	264	13
Cr	227	6
Cr pv	266	12
Cr sv	395	14
Mn	270	7
Mn pv	270	13
Mn sv	387	15
Fe	268	8
Fe pv	293	14
Fe sv	391	16
Co	268	9
Co pv	271	15
Co sv	390	17
Ni	270	10
Ni pv	368	16
Cu	295	11
Cu pv	369	17
Zn	277	12
Ga	135	3
Ga d	283	13
Ga h	405	13
Ge	174	4
Ge d	310	14
Ge h	410	14
As	209	5
As d	289	15
Se	212	6
Br	216	7
Kr	185	8
Rb pv	122	7
Rb sv	220	9
Sr sv	229	10
Y sv	203	11
Zr sv	230	12
Nb pv	209	11
Nb sv	293	13
Mo	225	6
Mo pv	225	12
Mo sv	243	14
Tc	229	7
Tc pv	264	13
Tc sv	319	15
Ru	213	8
Ru pv	240	14
Ru sv	319	16
Rh	229	9
Rh pv	247	15
Pd	251	10
Pd pv	251	16
Ag	250	11
Ag pv	298	17
Cd	274	12
In	96	3
In d	239	13
Sn	103	4
Sn d	241	14
Sb	172	5
Te	175	6
I	176	7
Xe	153	8
Cs sv	220	9
Ba sv	187	10
La	219	11
La s	137	9
Ce	273	12
Ce h	300	12
Ce 3	177	11
Pr	273	13
Pr 3	182	11
Nd	253	14
Nd 3	183	11
Pm	259	15
Pm 3	177	11
Sm	258	16
Sm 3	177	11
Eu	250	17
Eu 2	99	8
Eu 3	129	9
Gd	256	18
Gd 3	154	9
Tb	265	19
Tb 3	156	9
Dy	255	20
Dy 3	156	9
Ho	257	21
Ho 3	154	9
Er 2	120	8
Er 3	155	9
Er	298	22
Tm	257	23
Tm 3	149	9
Yb	253	24
Yb 2	113	8
Lu	256	25
Lu 3	155	9
Hf	220	4
Hf pv	220	10
Hf sv	237	12
Ta	224	5
Ta pv	224	11
W	223	6
W pv	223	12
Re	226	7
Re pv	226	13
Os	228	8
Os pv	228	14
Ir	211	9
Pt	230	10
Pt pv	295	16
Au	230	11
Hg	233	12
Tl	90	3
Tl d	237	13
Pb	98	4
Pb d	238	14
Bi	105	5
Bi d	243	15
Po	160	6
Po d	265	16
At	161	7
At d	266	17
Rn	152	8
Fr sv	215	9
Ra sv	237	10
Ac	172	11
Th	247	12
Th s	169	10
Pa	252	13
Pa s	193	11
U	253	14
U s	209	14
Np	254	15
Np s	208	15
Pu	254	16
Pu s	208	16
Am	256	17
Cm	258	18"""


# https://cms.mpi.univie.ac.at/vasp/vasp/Recommended_GW_PAW_potentials_vasp_5_2.html

# Element (and appendix)	default cutoff ENMAX (eV)	valency
gw_pot_text = """\
H GW	300	1
H h GW	700	1
He GW	405	2
Li sv GW	434	3
Li GW	112	1
Li AE GW	434	3
Be sv GW	537	4
Be GW	248	2
B GW	319	3
C GW	414	4
C GW new	414	4
C h GW	741	4
N GW	421	5
N GW new	421	5
N h GW	755	5
N s GW	296	5
O GW	415	6
O GW new	434	6
O h GW	765	6
O s GW	335	6
F GW	488	7
F GW new	488	7
F h GW	848	7
Ne GW	432	8
Ne s GW	318	8
Na sv GW	372	9
Mg sv GW	430	10
Mg GW	126	2
Mg pv GW	404	8
Al GW	240	3
Al sv GW	411	11
Si GW	245	4
Si GW new	245	4
Si sv GW	548	12
P GW	255	5
S GW	259	6
Cl GW	262	7
Ar GW	290	8
K sv GW	249	9
Ca sv GW	281	10
Sc sv GW	378	11
Ti sv GW	383	12
V sv GW	382	13
Cr sv GW	384	14
Mn sv GW	384	15
Mn GW	278	7
Fe sv GW	387	16
Fe GW	321	8
Co sv GW	387	17
Co GW	323	9
Ni sv GW	389	18
Ni GW	357	10
Cu pv GW	467	19
Cu sv GW	467	19
Cu GW	417	11
Zn sv GW	401	20
Zn GW	328	12
Ga d GW	404	13
Ga GW	135	3
Ga sv GW	404	21
Ge d GW	375	14
Ge sv GW	410	22
Ge GW	174	4
As GW	209	5
As sv GW	415	23
Se GW	212	6
Se sv GW	469	24
Br GW	216	7
Br sv GW	475	25
Kr GW	252	8
Rb sv GW	221	9
Sr sv GW	225	10
Y sv GW	339	11
Zr sv GW	346	12
Nb sv GW	353	13
Mo sv GW	344	14
Tc sv GW	351	15
Ru sv GW	348	16
Rh sv GW	351	17
Rh GW	247	9
Pd sv GW	356	18
Pd GW	251	10
Ag sv GW	354	19
Ag GW	250	11
Cd sv GW	361	20
Cd GW	254	12
In d GW	279	13
In sv GW	366	21
Sn d GW	260	14
Sn sv GW	368	22
Sb d GW	263	15
Sb sv GW	372	23
Sb GW	172	5
Te GW	175	6
Te sv GW	376	24
I GW	176	7
I sv GW	381	25
Xe GW	180	8
Xe sv GW	400	26
Cs sv GW	198	9
Ba sv GW	238	10
La GW	313	11
Ce GW	305	12
Hf sv GW	283	12
Ta sv GW	286	13
W sv GW	317	14
Re sv GW	317	15
Os sv GW	320	16
Ir sv GW	320	17
Pt sv GW	324	18
Pt GW	249	10
Au sv GW	306	19
Au GW	248	11
Hg sv GW	312	20
Tl d GW	237	15
Tl sv GW	316	21
Pb d GW	238	16
Pb sv GW	317	22
Bi d GW	261	17
Bi GW	147	5
Bi sv GW	323	23
Po d GW	267	18
Po sv GW	326	24
At d GW	266	17
At sv GW	328	25
Rn d GW	268	18
Rn sv GW	331	26"""


pot_dict = {
    'H': 'H_h_GW',
    'He': 'He_GW',
    'Li': 'Li_sv_GW',
    'Be': 'Be_sv_GW',
    'B': 'B_GW',
    'C': 'C_GW',
    'N': 'N_h_GW',
    'O': 'O_h_GW',
    'F': 'F_h_GW',
    'Ne': 'Ne_GW',
    'Na': 'Na_sv_GW',
    'Mg': 'Mg_sv_GW',
    'Al': 'Al_GW',
    'Si': 'Si_GW',
    'P': 'P_GW',
    'S': 'S_GW',
    'Cl': 'Cl_GW',
    'Ar': 'Ar_GW',
    'K': 'K_sv_GW',
    'Ca': 'Ca_sv_GW',
    'Sc': 'Sc_sv_GW',
    'Ti': 'Ti_sv_GW',
    'V': 'V_sv_GW',
    'Cr': 'Cr_sv_GW',
    'Mn': 'Mn_sv_GW',
    'Fe': 'Fe_sv_GW',
    'Co': 'Co_sv_GW',
    'Ni': 'Ni_sv_GW',
    'Cu': 'Cu_sv_GW',  # 'Cu_pv_GW',
    'Zn': 'Zn_sv_GW',
    'Ga': 'Ga_d_GW',
    'Ge': 'Ge_d_GW',
    'As': 'As_GW',
    'Se': 'Se_GW',
    'Br': 'Br_GW',
    'Kr': 'Kr_GW',
    'Rb': 'Rb_sv_GW',
    'Sr': 'Sr_sv_GW',
    'Y': 'Y_sv_GW',
    'Zr': 'Zr_sv_GW',
    'Nb': 'Nb_sv_GW',
    'Mo': 'Mo_sv_GW',
    'Tc': 'Tc_sv_GW',
    'Ru': 'Ru_sv_GW',
    'Rh': 'Rh_sv_GW',
    'Pd': 'Pd_sv_GW', # 'Pd_GW',
    'Ag': 'Ag_sv_GW', # 'Ag_GW',
    'Cd': 'Cd_sv_GW',
    'In': 'In_d_GW',
    'Sn': 'Sn_d_GW',
    'Sb': 'Sb_d_GW',
    'Te': 'Te_GW',
    'I': 'I_GW',
    'Xe': 'Xe_GW',
    'Cs': 'Cs_sv_GW',
    'Ba': 'Ba_sv_GW',
    'Hf': 'Hf_sv_GW',
    'Ta': 'Ta_sv_GW',
    'W': 'W_sv_GW',
    'Re': 'Re_sv_GW',
    'Os': 'Os_sv_GW',
    'Ir': 'Ir_sv_GW',
    'Pt': 'Pt_sv_GW',
    'Au': 'Au_sv_GW', # 'Au_pv_GW',
    'Pb': 'Pb_d', # 'Pb_d_GW'
    'Bi': 'Bi_d_GW'
}


def get_column_from_df(df, column, kpoint, encut, volume, volume_digits=6):
    selection = (df.n_kpts == kpoint) \
                & (df.encut == encut) \
                & (np.isclose(df.volume, volume, 10 ** -volume_digits))
    try:
        return df[selection][column].values[0]
    except IndexError:
        return None


def read_from_df(df, encut, kpoint, vol_space):
    vol_lst, eng_lst, conv_lst = zip(*[
        [
            get_column_from_df(df=df, column='volume', kpoint=kpoint, encut=encut, volume=vol_name, volume_digits=6),
            get_column_from_df(df=df, column='energy_tot', kpoint=kpoint, encut=encut, volume=vol_name, volume_digits=6),
            get_column_from_df(df=df, column='el_conv', kpoint=kpoint, encut=encut, volume=vol_name, volume_digits=6)
        ]
        for vol_name in vol_space
    ])
    vol_lst, eng_lst, conv_lst = np.array(vol_lst), np.array(eng_lst), np.array(conv_lst)
    return vol_lst, eng_lst, conv_lst


def get_alat_from_volume(volume, crystal_structure, number_of_atoms=1):
    if number_of_atoms == 1:
        if crystal_structure == 'bcc':
            return (2 * volume) ** (1/3)
        elif crystal_structure == 'fcc':
            return (4 * volume) ** (1/3)
        else:
            raise ValueError('unknown crystal_structure')
    else:
        return volume ** (1/3)


def lattice_constants_for_element_ase(element):
    if '_' in element:
        element = element.split('_')[0]  # Convert VASP potential name to chemical element
    ref = reference_states[atomic_numbers[element]]
    alat_base = ref['a']
    return ref['symmetry'], alat_base


def get_single_column_from_df(df, column, volume, volume_digits=6):
    selection = (np.isclose(df.volume, volume, 10 ** -volume_digits))
    try:
        return df[selection][column].values[0]
    except IndexError:
        return np.nan


def calc_v0_from_fit_funct(fit_funct, x):
    fit_funct_der = fit_funct.deriv().r
    fit_funct_der_r = fit_funct_der[fit_funct_der.imag == 0].real
    fit_funct_der_val = fit_funct.deriv(2)(fit_funct_der_r)
    select = (fit_funct_der_val > 0) & (fit_funct_der_r > np.min(x)) & (fit_funct_der_r < np.max(x))
    v0_lst = fit_funct_der_r[select]
    if len(v0_lst) == 1:
        return v0_lst[0]
    else:
        select = fit_funct_der_val > 0
        v0_lst = fit_funct_der_r[select]
        if len(v0_lst) == 1:
            return v0_lst[0]
        else:
            return np.nan


def bulk_modulus_from_fit(fit_funct, x, convert_to_gpa=True):
    v_0 = calc_v0_from_fit_funct(fit_funct=fit_funct, x=x)
    return bulk_modulus_from_fit_and_volume(fit_funct=fit_funct, v_0=v_0, convert_to_gpa=convert_to_gpa)


def bulk_modulus_from_fit_and_volume(fit_funct, v_0, convert_to_gpa=True):
    if v_0 is not np.nan:
        bulk = np.polyder(fit_funct, 2)(v_0) * v_0
        if convert_to_gpa:
            return bulk * eV_div_A3_to_GPa
        else:
            return bulk
    else:
        return np.nan


def bulk_modulus_dereivative_from_fit(fit_funct, x):
    v_0 = calc_v0_from_fit_funct(fit_funct=fit_funct, x=x)
    return bulk_modulus_dereivative_from_fit_and_volume(fit_funct=fit_funct, v_0=v_0)


def bulk_modulus_dereivative_from_fit_and_volume(fit_funct, v_0):
    if v_0 is not np.nan:
        return -1 + - v_0 * np.polyder(fit_funct, 3)(v_0) / np.polyder(fit_funct, 2)(v_0)
    else:
        return np.nan


def get_line(limit, encut_space, mat):
    return [np.min(lst) if len(lst) > 0 else np.max(encut_space) for lst in [encut_space[noise<limit] for noise in mat]]


def shift_lst(lst):
    res = []
    for i in lst[::-1]:
        if len(res) > 0 and res[-1] > i:
            res.append(res[-1])
        else:
            res.append(i)
    return res[::-1]


def double_smooth(mat):
    return np.array([shift_lst(lst=lst) for lst in np.array([shift_lst(lst=lst) for lst in mat]).T]).T


def get_potential_encut(el, default_potential):
    s = Settings()
    vp = VaspPotential()
    df_pot = vp.pbe.find(element=el)
    encut_low = float([
        s_path for s_path in s.resource_paths if "resources" in s_path
    ][0] + "/vasp/potentials/" + df_pot[df_pot.Name == default_potential + "-gga-pbe"].ENMAX.values[0])
    return encut_low


def get_delta_project_recommendation():
    d_el_lst, d_en_lst, d_k_lst = zip(*[l.split() for l in delta_project_recommendations.split("\n")])
    d_en_lst = [int(e) for e in d_en_lst]
    d_k_lst = [int(k) for k in d_k_lst]
    return pandas.DataFrame({"element": d_el_lst, "encut": d_en_lst, "kpoint": d_k_lst})


def get_conv_parameter(element, mprester):
    data = mprester.get_data(element)
    mp_id = [d['material_id'] for d in data if d['spacegroup']['symbol']=='Fm-3m' and int(d['unit_cell_formula'][element])==1][0]
    data_input = mprester.query(mp_id, ['input.incar', 'input.kpoints'])
    k = data_input[0]['input.kpoints'].as_dict()['kpoints'][0][0]
    e = data_input[0]['input.incar']['ENCUT']
    return e, k


def get_materials_project_recommendation(element_lst, token):
    m = MPRester(token)
    mp_e_lst, mp_k_lst = zip(*[get_conv_parameter(element=e, mprester=m) for e in element_lst])
    return pandas.DataFrame({"element": element_lst, "encut": mp_e_lst, "kpoint": mp_k_lst})


def collect_data(file_pre, file, el, encut_space, kpoint_space, uncertainty_parameter, degree, data_path):
    pr_pre = Project(".")
    structure_base = pr_pre.create_ase_bulk(el)
    vol_eq_pre = structure_base.get_volume()
    vol_range = np.linspace(
        (1 - float(uncertainty_parameter['vol_range'])) * vol_eq_pre,
        (1 + float(uncertainty_parameter['vol_range'])) * vol_eq_pre,
        int(uncertainty_parameter['points'])
    )
    calc_df_pre = pandas.read_csv(data_path + file_pre, index_col=0)
    vol_lst, eng_lst, conv_lst = read_from_df(
        df=calc_df_pre,
        encut=encut_space[-1],
        kpoint=kpoint_space[-1],
        vol_space=vol_range
    )
    vol_lst, eng_lst = zip(*[[vol, eng] for vol, eng in zip(vol_lst, eng_lst) if eng is not None])
    fit_pre = np.poly1d(np.polyfit(vol_lst, eng_lst, degree))
    vol_eq = np.round(calc_v0_from_fit_funct(fit_funct=fit_pre, x=vol_lst), 4)
    print(el, vol_eq)
    vol_range = np.linspace(
        (1 - float(uncertainty_parameter['vol_range'])) * vol_eq,
        (1 + float(uncertainty_parameter['vol_range'])) * vol_eq,
        int(uncertainty_parameter['points'])
    )
    default_potential = file[:-10]
    encut_low = get_potential_encut(el=el, default_potential=default_potential)
    encut_low = np.min(encut_space[encut_space >= encut_low])
    encut_low_ind = np.min(np.arange(len(encut_space))[encut_space >= encut_low])
    kpoint_low_ind = 4
    kpoint_low = kpoint_space[kpoint_low_ind]
    print(default_potential, encut_low, kpoint_low, encut_low_ind, kpoint_low_ind)
    calc_df = pandas.read_csv(data_path + file, index_col=0)
    eng_encut_low_lst, eng_encut_high_lst, eng_kpoint_low_lst, eng_kpoint_high_lst = [], [], [], []
    for encut in encut_space:
        df_tmp = calc_df[(calc_df.n_kpts == kpoint_low) & (calc_df.encut == encut)]
        eng_lst = [
            get_single_column_from_df(
                df=df_tmp,
                column='energy_tot',
                volume=vol_name,
                volume_digits=6
            )
            for vol_name in vol_range
        ]
        eng_encut_low_lst.append(eng_lst)
    for encut in encut_space:
        df_tmp = calc_df[(calc_df.n_kpts == kpoint_space[-1]) & (calc_df.encut == encut)]
        eng_lst = [
            get_single_column_from_df(
                df=df_tmp,
                column='energy_tot',
                volume=vol_name,
                volume_digits=6
            )
            for vol_name in vol_range
        ]
        eng_encut_high_lst.append(eng_lst)
    for kpoint in kpoint_space:
        df_tmp = calc_df[(calc_df.n_kpts == kpoint) & (calc_df.encut == encut_low)]
        eng_lst = [
            get_single_column_from_df(
                df=df_tmp,
                column='energy_tot',
                volume=vol_name,
                volume_digits=6
            )
            for vol_name in vol_range
        ]
        eng_kpoint_low_lst.append(eng_lst)
    for kpoint in kpoint_space:
        df_tmp = calc_df[(calc_df.n_kpts == kpoint) & (calc_df.encut == encut_space[-1])]
        eng_lst = [
            get_single_column_from_df(
                df=df_tmp,
                column='energy_tot',
                volume=vol_name,
                volume_digits=6
            )
            for vol_name in vol_range
        ]
        eng_kpoint_high_lst.append(eng_lst)
    eng_encut_low_lst, eng_encut_high_lst, eng_kpoint_low_lst, eng_kpoint_high_lst = np.array(
        eng_encut_low_lst), np.array(eng_encut_high_lst), np.array(eng_kpoint_low_lst), np.array(eng_kpoint_high_lst)

    ediff_kpoint_lst, ediff_encut_lst = [], []
    for j, kpoint in enumerate(kpoint_space):
        vol_lst, ediff = np.array([[vo, en] for vo, en in zip(vol_range,
                                                              eng_kpoint_low_lst[-1] - eng_kpoint_low_lst[j] - (
                                                                          eng_kpoint_high_lst[-1] - eng_kpoint_high_lst[
                                                                      j])) if not np.isnan(en)]).T
        ediff_kpoint_lst.append(np.std(ediff))

    for i, encut in enumerate(encut_space):
        vol_lst, ediff = np.array([[vo, en] for vo, en in zip(vol_range,
                                                              eng_encut_low_lst[-1] - eng_encut_low_lst[i] - (
                                                                          eng_encut_high_lst[-1] - eng_encut_high_lst[
                                                                      i])) if not np.isnan(en)]).T
        ediff_encut_lst.append(np.std(ediff))

    fit_base = np.poly1d(np.polyfit(vol_range, eng_encut_low_lst[encut_low_ind], degree))
    base_flux = fit_base(vol_range) - eng_encut_low_lst[encut_low_ind]
    ediff_base = np.std(base_flux)

    encut_rec = encut_low_ind
    kpoint_rec = kpoint_low_ind

    eng_sys_encut_lst = []
    for i, encut in enumerate(encut_space):
        eng_sys_encut_lst.append(eng_encut_high_lst[-1] - eng_encut_high_lst[i])
    eng_sys_kpoint_lst = []
    for j, kpoint in enumerate(kpoint_space):
        eng_sys_kpoint_lst.append(eng_kpoint_high_lst[-1] - eng_kpoint_high_lst[j])

    ediff_total_lst, eng_total_lst, b0_sys_total_lst, b0_stat_total_lst = [], [], [], []
    for i, encut in enumerate(encut_space):
        ediff_tmp_lst, eng_tmp_lst, b0_sys_tmp_lst, b0_stat_tmp_lst = [], [], [], []
        for j, kpoint in enumerate(kpoint_space):
            ediff_tmp = ediff_base * ediff_encut_lst[i] / ediff_encut_lst[encut_rec] * ediff_kpoint_lst[j] / \
                        ediff_kpoint_lst[kpoint_rec]
            ediff_tmp_lst.append(ediff_tmp)
            eng_tmp = eng_encut_high_lst[-1] + eng_sys_encut_lst[i] + eng_sys_kpoint_lst[j]
            eng_tmp_lst.append(eng_tmp)
            vol_lst, eng_tmp = np.array([[vo, en] for vo, en in zip(vol_range, eng_tmp) if not np.isnan(en)]).T
            fit_base = np.poly1d(np.polyfit(vol_lst, eng_tmp, degree))
            v0_tmp = calc_v0_from_fit_funct(fit_funct=fit_base, x=vol_range)
            b0_tmp = bulk_modulus_from_fit_and_volume(fit_funct=fit_base, v_0=v0_tmp, convert_to_gpa=True)
            b0_sys_tmp_lst.append(b0_tmp)
            b0_stat_lst = []
            for t in range(100):
                eng_test = fit_base(vol_range) + np.random.normal(loc=0.0, scale=ediff_tmp, size=len(vol_range))
                fit_tmp = np.poly1d(np.polyfit(vol_range, eng_test, degree))
                v0_tmp = calc_v0_from_fit_funct(fit_funct=fit_tmp, x=vol_range)
                b0_tmp = bulk_modulus_from_fit_and_volume(fit_funct=fit_tmp, v_0=v0_tmp, convert_to_gpa=True)
                b0_stat_lst.append(b0_tmp)
            b0_stat_tmp_lst.append(np.std(b0_stat_lst))
        ediff_total_lst.append(ediff_tmp_lst)
        eng_total_lst.append(eng_tmp_lst)
        b0_sys_total_lst.append(b0_sys_tmp_lst)
        b0_stat_total_lst.append(b0_stat_tmp_lst)

    b0_stat_total_lst = double_smooth(b0_stat_total_lst)
    b0_approx_diff_mat = double_smooth(np.abs(b0_sys_total_lst - np.array(b0_sys_total_lst)[-1, -1]))
    b0_approx_diff_mat_ns = np.abs(b0_sys_total_lst - np.array(b0_sys_total_lst)[-1, -1])

    return encut_low, b0_stat_total_lst, b0_sys_total_lst, b0_approx_diff_mat, b0_approx_diff_mat_ns


def filter_job_type(job):
    return job.__name__ == 'Vasp'


def wait_for_jobs(project, job_lst, interval_in_s=10, max_iterations=10000):
    for job in job_lst:
        project.wait_for_job(job, interval_in_s=interval_in_s, max_iterations=max_iterations)


def get_potential(element=None, potential_file=None, project=None):
    if project is None:
        project = Project('.')
    if potential_file is None and element is not None:
        potential_file = pot_dict[element]
    elif potential_file is not None and element is None:
        element = potential_file.split('_')[0]
    else:
        raise ValueError()
    return project.create_element(
        new_element_name=potential_file,
        parent_element=element,
        potential_file=potential_file
    )


class EnCutRecommended(object):
    def __init__(self):
        self._encut_dict = self._generate_encut_dict()
        self._vasp_potential = VaspPotential()

    @staticmethod
    def _conv_line_fucnt(l):
        l_spilt = l.split('\t')
        name = l_spilt[0].replace(' ', '_')
        encut = int(l_spilt[1])
        return name, encut

    def _conv_encut_dict(self, text_input):
        return {name: encut for name, encut in [self._conv_line_fucnt(l) for l in text_input.split('\n')]}

    def _generate_encut_dict(self):
        return {**self._conv_encut_dict(text_input=basic_pot_text),
                **self._conv_encut_dict(text_input=gw_pot_text)}  # only python 3.5

    def find_encut_recommended(self, element=None, element_str=None, xc='pbe'):
        """
        Look up the default Energy Cutoff based on:
        https://cms.mpi.univie.ac.at/vasp/vasp/Recommended_GW_PAW_potentials_vasp_5_2.html

        Args:
            element (str): chemical element ["Ca", ...]
            element_str (str): element string in VASP to define the pseudo potential ["Ca_sv", ...]
            xc (str): exchange correlation functional ["pbe", "lda"]

        Returns:
            int: recommended Energy Cutoff
        """
        if element_str is None:
            if xc.lower() == 'pbe':
                element_str = self._vasp_potential.pbe.find_default(element)['Filename'].values[0][0].split('/')[1]
            elif xc.lower() == 'lda':
                element_str = self._vasp_potential.lda.find_default(element)['Filename'].values[0][0].split('/')[1]
            else:
                raise ValueError('Unsupported exchange correlation functional xc: ["pbe", "lda"]')
        else:
            if element is not None:
                raise ValueError('Either set the element or the element_str, both is not possible.')
        if element_str in self._encut_dict.keys():
            return self._encut_dict[element_str]
        else:
            return self._encut_dict[element_str.split("_")[0]]


def find_recommended_encut(element=None, element_str=None, xc='pbe'):
    """
    Look up the default Energy Cutoff based on:
    https://cms.mpi.univie.ac.at/vasp/vasp/Recommended_GW_PAW_potentials_vasp_5_2.html

    Args:
        element (str): chemical element ["Ca", ...]
        element_str (str): element string in VASP to define the pseudo potential ["Ca_sv", ...]
        xc (str): exchange correlation functional ["pbe", "lda"]

    Returns:
        int: recommended Energy Cutoff
    """
    return EnCutRecommended().find_encut_recommended(element=element, element_str=element_str, xc=xc)


def job_name_funct(volume, encut, kpoints, digits=6):
    job_name = "job__{:.6f}__{:.6f}__".format(np.round(volume, digits), np.round(encut, digits)) + str(kpoints)
    job_name = job_name.replace('.', '_')
    return job_name


def setup_calculation(pr, structure, encut, kpoints, uncertainty_parameter):
    electronic_energy = uncertainty_parameter['electronic_convergence']
    queue = uncertainty_parameter['queue']
    run_time = uncertainty_parameter['run_time']
    cores = uncertainty_parameter['cores']
    memory_factor = uncertainty_parameter['memory_factor']

    job_name = job_name_funct(structure.get_volume() / len(structure), encut, kpoints, digits=6)
    job = pr.create_job(pr.job_type.Vasp, job_name)
    if queue is not None:
        job.server.queue = queue

    # Vasp settings
    job.structure = structure
    job.set_encut(encut)
    job.set_kpoints([kpoints, kpoints, kpoints])

    job.input.incar['SIGMA'] = 0.2
    job.input.incar['ISMEAR'] = -5

    job.input.incar['ALGO'] = 'normal'  # somehow the fast algorithm fails for Ca and Sr
    job.input.incar['LASPH'] = True  # include non-spherical contributions - recommended by the delta project
    job.input.incar['NELM'] = 120
    job.input.incar['NEDOS'] = 501  # recommendation from Kurt Lejaeghere
    job.set_convergence_precision(electronic_energy=electronic_energy)

    job.server.run_time = run_time
    # job.input.incar['NCORE'] = int(np.sqrt(cores))  # only required for larger calculations
    job.server.cores = cores
    job.server.memory_limit = str(int(3 * memory_factor * cores)) + 'GB'
    if '5.4.4_bl' in job.executable.list_executables():
        job.executable.version = '5.4.4_bl'  # use patched version

    job.write_charge_density = False
    job.write_wave_funct = False
    return job


def get_strain(job):
    return float(job.job_name.split('__')[1].replace('_', '.'))


def get_e_conv_level(job):
    return np.max(np.abs(
        job['output/generic/dft/scf_energy_free'][0] -
        job['output/generic/dft/scf_energy_free'][0][-1]
    )[-10:])


def setup_pyiron_table(project):
    pyiron_table = project.create_job(project.job_type.TableJob, 'table')
    pyiron_table.analysis_project = project
    pyiron_table.filter_function = filter_job_type
    _ = pyiron_table.add.get_elements
    _ = pyiron_table.add.get_energy_tot_per_atom
    _ = pyiron_table.add.get_job_name
    _ = pyiron_table.add.get_sigma
    _ = pyiron_table.add.get_encut
    _ = pyiron_table.add.get_n_kpts
    _ = pyiron_table.add.get_n_equ_kpts
    _ = pyiron_table.add.get_ismear
    _ = pyiron_table.add.get_average_waves
    _ = pyiron_table.add.get_ekin_error
    _ = pyiron_table.add.get_volume_per_atom
    pyiron_table.add['el_conv'] = get_e_conv_level
    # pyiron_table.add['strain'] = get_strain
    pyiron_table.run()
    pyiron_table.update_table(job_status_list=["finished", "not_converged"])
    return pyiron_table


def get_alat_from_structure(structure, crystal_structure):
    return get_alat_from_volume(
        volume=structure.get_volume(),
        crystal_structure=crystal_structure,
        number_of_atoms=len(number_of_atoms)
    )


def get_alat_range(vol_eq, strain, steps, crystal_structure, number_of_atoms=1):
    alat_min = get_alat_from_volume(
        volume=(1-strain) * vol_eq,
        crystal_structure=crystal_structure,
        number_of_atoms=number_of_atoms
    )
    alat_max = get_alat_from_volume(
        volume=(1+strain) * vol_eq,
        crystal_structure=crystal_structure,
        number_of_atoms=number_of_atoms
    )
    return np.linspace(alat_min, alat_max, steps)


def create_structure(pr, alat, uncertainty_parameter):
    if uncertainty_parameter['crystal_structure'].lower() == 'bcc':
        structure = pr.create_ase_bulk('Fe', a=alat)
        el = get_potential(
            element=None,
            potential_file=uncertainty_parameter['pseudo_potential'],
            project=pr)
        structure[:] = el
    elif uncertainty_parameter['crystal_structure'].lower() == 'fcc':
        structure = pr.create_ase_bulk('Al', a=alat)
        el = get_potential(
            element=None,
            potential_file=uncertainty_parameter['pseudo_potential'],
            project=pr)
        structure[:] = el
    else:
        raise ValueError('unknown crystal_structure')
    return structure


def get_bound(lst):
    new_lst = []
    n_prev = None
    for n in lst[::-1]:
        if n_prev is not None and n_prev > n:
            new_lst.append(n_prev)
        else:
            new_lst.append(n)
            n_prev = n
    return new_lst[::-1]


def get_eq_parameter(fit_tmp, vol_tmp_lst):
    v0_tmp = calc_v0_from_fit_funct(fit_funct=fit_tmp, x=vol_tmp_lst)
    b0_tmp = bulk_modulus_from_fit_and_volume(fit_funct=fit_tmp, v_0=v0_tmp, convert_to_gpa=True)
    bp_tmp = bulk_modulus_dereivative_from_fit_and_volume(fit_funct=fit_tmp, v_0=v0_tmp)
    return v0_tmp, b0_tmp, bp_tmp


def wait_for_jobs_to_be_done(project, sleep_period=30, iteration=10000):
    for i in range(iteration):
        df = project.job_table()
        if len(df[(df.status == 'running') | (df.status == 'submitted')]) > 0:
            time.sleep(sleep_period)
        else:
            break


def calc_set_of_jobs(pr, parameter_lst, uncertainty_parameter, sleep_period=30, iteration=10000):
    df_job_table = pr.job_table()
    if len(df_job_table) == 0:
        job_name_lst = []
    else:
        job_name_lst = df_job_table.job.values.tolist()
    for p in parameter_lst:
        alat, encut, kpoints = p
        structure = create_structure(
            pr=pr,
            alat=alat,
            uncertainty_parameter=uncertainty_parameter
        )
        job_name = job_name_funct(structure.get_volume()/ len(structure), encut, kpoints, digits=6)
        if job_name not in job_name_lst:
            # print('create new job!')
            job = setup_calculation(
                pr=pr,
                structure=structure,
                encut=encut,
                kpoints=kpoints,
                uncertainty_parameter=uncertainty_parameter
            )
            job.run()
    wait_for_jobs_to_be_done(project=pr, sleep_period=sleep_period, iteration=iteration)


def run_calc_alat(pr, alat_lst, encut, kpoints, uncertainty_parameter, sleep_period=30, iteration=10000, pytab=None):
    parameter_lst = [[alat, encut, kpoints] for alat in alat_lst]
    calc_set_of_jobs(
        pr=pr,
        parameter_lst=parameter_lst,
        uncertainty_parameter=uncertainty_parameter,
        sleep_period=sleep_period,
        iteration=iteration
    )
    if pytab is None:
        pytab = setup_pyiron_table(project=pr)
    else:
        pytab.update_table()
    return pytab


def remove_none(lst):
    return [l for l in lst if l is not None]


