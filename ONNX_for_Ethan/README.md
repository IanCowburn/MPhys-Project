# Hi Ethan 🐔

## Transformer Dataloading information:

### 1. Loading the files and considering variables

Files used:

files = ["tttt_NLO_523243_mc23a_fullsim.root",
         "tttt_NLO_523243_mc23d_fullsim.root",
         "tttt_NLO_523243_mc23e_fullsim.root"]

Variables used:

var_names = ["lepton_eta", "lepton_phi", "jet_eta", "jet_phi", "lepton_pt_NOSYS", "jet_pt_NOSYS", "lepton_e_NOSYS", "jet_e_NOSYS", "met_met_NOSYS", "met_phi_NOSYS", "jet_GN2v01_FixedCutBEff_77_select", "lepton_charge"]

### 2. Selecting events

Only consider events with up to two (charged) leptons and up to 12 jets.

#### This means you get 15 tokens:

##### 2 Lepton tokens:

(Eta, Phi, Pt, E, Lepton charge, B-jet tagging (i.e. 0))

##### 12 Jet tokens:

(Eta, Phi, Pt, E, Lepton charge (i.e. 0), B-jet tagging)

##### 1 MET token:

(Eta (i.e. 0), MET Phi, MET MET, E (i.e. 0), Lepton charge (i.e. 0), B-jet tagging (i.e. 0))

#### Then pad the arrays:

MET variables aren't padded as they're event level scalars.

Pad all the other missing values with -99 (and remember the padding for each event).

Assemble the data by concatenating lepton and jet arrays. This means you just combine all the first four features and then for the remaining two add the zeros for the leptons for b-tagging and the zeros for jets for charge, something like this:

        data_array = np.concatenate([
            lepton_arrays[:4, :, :],  # eta, phi, pt, e for leptons
            jet_arrays[:4, :, :]      # eta, phi, pt, e for jets
        ], axis=2)
        
        # Add charge feature (leptons have charge, jets have 0)
        charge_zeros = np.zeros([num_events, self.jet_mask_size])
        charge_leptons = np.concatenate([lepton_arrays[4, :, :], charge_zeros], axis=1)  # (events, total_tokens)
        
        # Add b-jet tag feature (leptons have 0, jets have tagging)
        bjet_zeros = np.zeros([num_events, self.lepton_mask_size])
        bjet_tags = np.concatenate([bjet_zeros, jet_arrays[4, :, :]], axis=1)  # (events, total_tokens)
