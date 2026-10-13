# CAPPIBARAS_Jerry
Files I added for CAPPIBARAS, Wang2026.py goes into Models/Papers, Wang2026.yaml goes into yamls, plot_Wang2026 and submit_Wang2026 goes into CAPPIBARAS-main.

Additional changes I made to CAPPIBARAS original files:

1. ForwardModel.py:

#prof_move = self.ave_hod(prof)
prof_Mave = self.ave_Mh(prof)


2. runchains.py:

default_mcmc = {
    "Rminus1_stop": 0.01,
    "max_tries": "200d",
    "output_every": "10s",
    "learn_every": 40,
}

yaml_info.setdefault("sampler", {})
yaml_info["sampler"].setdefault("mcmc", {})

for key, value in default_mcmc.items():
    yaml_info["sampler"]["mcmc"].setdefault(key, value)



3. SHMRs.py in OLDMODULES

from Models.OldModules import Studies




4. Popik2026.py

P0=self.PL(p['P0'], p['P0_z'], p['P0_m'])
xc=self.PL(p['xc_P'], p['xc_P_z'], p['xc_P_m'])
gamma=self.PL(p['gamma_P'], p['gamma_P_z'], p['gamma_P_m'])
alpha=self.PL(p['alpha_P'], p['alpha_P_z'], p['alpha_P_m'])
beta=self.PL(p['beta_P'], p['beta_P_z'], p['beta_P_m'])

#P0 = p['P0']
#xc = p['xc_P']
#gamma = p['gamma_P']
#alpha = p['alpha_P']
#beta = p['beta_P']




