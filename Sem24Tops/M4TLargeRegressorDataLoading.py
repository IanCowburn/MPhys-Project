
# Imports
import uproot
import awkward as ak
import vector
import numpy as np
import os
# Data files
files = ["4topLO_24March26_minus.root", "4topLO_24March26_plus.root"]

var_names = ['el_eta', 'el_phi', 'el_pt', 'el_charge',
             'mu_eta', 'mu_phi', 'mu_pt', 'mu_charge',
             'jet_eta', 'jet_phi', 'jet_pt', 'jet_mass', 'jet_btag',
             'met_phi', 'met_met']

class TransformerDataLoader:
    def __init__(self, files, var_names, lepton_mask_size, jet_mask_size):
        """
        Initialises the data loader with file paths, variable names, and mask sizes.
        Args:
            files (list): List of ROOT file paths.
            var_names (list): List of variable names to extract from the ROOT files.
            lepton_mask_size (int): Maximum number of leptons allowed per event.
            jet_mask_size (int): Maximum number of jets allowed per event.
        Returns:
            None
        """
    
        self.files = files
        self.var_names = var_names
        self.lepton_mask_size = lepton_mask_size
        self.jet_mask_size = jet_mask_size
        self.used_var_names = ['lepton_eta', 'lepton_phi', 'lepton_pt', 'lepton_e', 'lepton_charge',
             'jet_eta', 'jet_phi', 'jet_pt', 'jet_e', 'jet_btag',
             'met_phi', 'met_met',
             'nJets', 'nBjets', 'nElectrons', 'nMuons',
             'HT_all', 'HT_jets']
        
    def data_loading(self, filenames):
        """
        Initially opens the root file and extracts the reco tree. Creates parton 4-momentum vector from
        the tree branches. Then extracts the relevant variable name branches from the tree into the
        main data array.
        Args:
            self
            filenames (str): Path to the ROOT file.
        Returns:
            tuple:
                data_array (awkward.Array): Main data array with variable branches.
                parton_vectors (vector.Array): Parton-level 4-momentum vectors.
                tree (uproot.Tree): The ROOT tree containing the reco data.
        """
        file_tt = uproot.open(filenames)
        tree = file_tt["truth"]

        parton_vectors = vector.zip({
            "pt": tree["top_pt"].array(),
            "eta": tree["top_eta"].array(),
            "phi": tree["top_phi"].array(),
            "mass": tree["top_mass"].array()
        })

        parton_pt = tree["top_pt"].array()

        tree = file_tt["reco"]

        data_array = tree.arrays(self.var_names)
        

        data_array["lepton_eta"] = ak.concatenate([data_array['el_eta'], data_array['mu_eta']], axis=1)
        data_array["lepton_phi"] = ak.concatenate([data_array['el_phi'], data_array['mu_phi']], axis=1)
        data_array["lepton_pt"] = ak.concatenate([data_array['el_pt'], data_array['mu_pt']], axis=1)
        data_array["lepton_charge"] = ak.concatenate([data_array['el_charge'], data_array['mu_charge']], axis=1)

        electron_mass = 0.000511 # GeV
        muon_mass = 0.10566 # GeV
        data_array["electron_mass"] = ak.fill_none(ak.broadcast_arrays(electron_mass, data_array['el_pt'])[0], 0)
        data_array["muon_mass"] = ak.fill_none(ak.broadcast_arrays(muon_mass, data_array['mu_pt'])[0], 0)
        data_array["lepton_mass"] = ak.concatenate([data_array["electron_mass"], data_array["muon_mass"]], axis=1)

        jet_vector = vector.zip({
            "pt": data_array["jet_pt"],
            "eta": data_array["jet_eta"],
            "phi": data_array["jet_phi"],
            "mass": data_array["jet_mass"]
        })
        data_array["jet_e"] = jet_vector.e

        lepton_vector = vector.zip({
            "pt": data_array["lepton_pt"],
            "eta": data_array["lepton_eta"],
            "phi": data_array["lepton_phi"],
            "mass": data_array["lepton_mass"]
        })
        data_array["lepton_e"] = lepton_vector.e

        data_array["nJets"] = ak.num(data_array["jet_eta"])
        data_array["nElectrons"] = ak.num(data_array["el_eta"])
        data_array["nMuons"] = ak.num(data_array["mu_eta"])
        data_array["nBjets"] = ak.sum(data_array["jet_btag"] == 1, axis=1)

        data_array["HT_all"] = ak.sum(np.abs(data_array["jet_pt"]), axis=1) + ak.sum(np.abs(data_array["lepton_pt"]), axis=1)
        data_array["HT_jets"] = ak.sum(np.abs(data_array["jet_pt"]), axis=1)
        
        return data_array, parton_vectors, tree, parton_pt
    
    def initial_masking(self, data_array, parton_vectors, tree, parton_pt):
        """
        Takes in the data array, the parton vectors, and the tree from the ROOT file. Also takes in the
        lepton and jet mask sizes to filter out events with too many leptons or jets.
        Creates masking arrays for the leptons and jets based on their counts in each event. The main
        data array and parton vectors are then filtered using these masks to only include events that
        meet the jet and lepton masks.
        The combined parton system is created here after padding.
        The first for loop determines the padding sizes for each variable in var_names by finding the maximum
        number of entries across all events. This is stored in the padding_size list for each variable.
        The second for loop applies padding to each variable in var_names using the previously determined
        padding sizes. It uses ak.pad_none to pad the arrays to the specified size, and then fills any
        missing values with -99 using ak.fill_none.
        Args:
            data_array (awkward.Array): Input data array with variable branches.
            parton_vectors (vector.Array): Parton-level 4-momentum vectors.
            tree (uproot.Tree): The ROOT tree containing the reco data.
            lepton_mask_size (int): Maximum number of leptons allowed per event.
            jet_mask_size (int): Maximum number of jets allowed per event.
            var_names (list): List of variable names to process.
        Returns:
            tuple:
                data_array (awkward.Array): Padded and filtered data array.
                combined_parton_system (vector.Array): Combined parton system from the 4-momenta.
        """
            
        masking_array = data_array[["jet_eta", "lepton_eta"]]
        nLeptons_mask = ak.num(masking_array["lepton_eta"]) <= self.lepton_mask_size
        nJets_mask = ak.num(masking_array["jet_eta"]) <= self.jet_mask_size
        data_array = data_array[nLeptons_mask & nJets_mask]
        parton_vectors = parton_vectors[nLeptons_mask & nJets_mask]
        combined_parton_system = parton_vectors[:,0] + parton_vectors[:,1] + parton_vectors[:,2] + parton_vectors[:,3]
        parton_pt = parton_pt[nLeptons_mask & nJets_mask]
        parton_ht = ak.sum(parton_pt, axis=1)

        # Deterministic padding by token type to guarantee consistent shapes across features.
        for name in self.used_var_names:
            if name.startswith("lepton_"):
                pad = self.lepton_mask_size
            elif name.startswith("jet_"):
                pad = self.jet_mask_size
            else:
                continue

            data_array[name] = ak.pad_none(data_array[name], pad, clip=True)
            data_array[name] = ak.fill_none(data_array[name], -99)
        
        return data_array, combined_parton_system, parton_ht
    
    def data_combination(self, data_array, combined_parton_system, parton_ht):
        """
        Creates the final data array by concatenating lepton and jet arrays along the feature axis.
        It first separates the lepton and jet related arrays from the main data array based on the variable names.
        Then, it converts these awkward arrays to numpy arrays for easier manipulation.
        Finally, it concatenates the lepton and jet arrays along the feature axis and transposes the resulting array
        to have the shape (events, tokens, features).
        Args:
            data_array (awkward.Array): Input data array with variable branches.
            combined_parton_system (vector.Array): Combined parton system from the 4-momenta.
        Returns:
            tuple:
                data_array (numpy.ndarray): Final combined data array with shape (events, tokens, features).
                combined_parton_system (vector.Array): Combined parton system from the 4-momenta.
        """
        lepton_arrays = []
        jet_arrays = []
        met_arrays = []
        numbers_arrays = []
        ht_arrays = []
        for name in self.used_var_names:
            if name.startswith("lepton_"):
                lepton_arrays.append(data_array[name])
            elif name.startswith("jet_"):
                jet_arrays.append(data_array[name])
            elif name.startswith("met_"):
                met_arrays.append(data_array[name])
            elif name in ["nJets", "nBjets", "nElectrons", "nMuons"]:
                numbers_arrays.append(data_array[name])
            elif name.startswith("HT_"):
                ht_arrays.append(data_array[name])
        
        # Convert each branch separately to avoid irreducible UnionArray conversion errors.
        lepton_arrays = np.stack([
            ak.to_numpy(ak.values_astype(arr, np.float32)) for arr in lepton_arrays
        ], axis=0)  # (5, events, lepton_tokens)
        jet_arrays = np.stack([
            ak.to_numpy(ak.values_astype(arr, np.float32)) for arr in jet_arrays
        ], axis=0)  # (5, events, jet_tokens)
        met_arrays = np.stack([
            ak.to_numpy(ak.values_astype(arr, np.float32)) for arr in met_arrays
        ], axis=0)  # (2, events)
        numbers_arrays = np.stack([
            ak.to_numpy(ak.values_astype(arr, np.float32)) for arr in numbers_arrays
        ], axis=0)  # (4, events)
        ht_arrays = np.stack([
            ak.to_numpy(ak.values_astype(arr, np.float32)) for arr in ht_arrays
        ], axis=0)  # (2, events)

        met_arrays = np.squeeze(met_arrays)
        met_arrays = np.concatenate([met_arrays, np.zeros((3, met_arrays.shape[1]))], axis=0) # Pad to 5 features
        numbers_arrays = np.concatenate([numbers_arrays, np.zeros((1, numbers_arrays.shape[1]))], axis=0) # Pad to 5 features
        ht_arrays = np.concatenate([ht_arrays, np.zeros((3, ht_arrays.shape[1]))], axis=0) # Pad to 5 features
        
        data_array = np.concatenate([
            lepton_arrays,
            jet_arrays,
            met_arrays[:, :, np.newaxis],  # Add new axis for concatenation
            numbers_arrays[:, :, np.newaxis],  # Add new axis for concatenation
            ht_arrays[:, :, np.newaxis]  # Add new axis for concatenation
        ], axis=2)

        # Transpose to (events, 17, 5)
        data_array = np.transpose(data_array, (1, 2, 0))
        
        return data_array, combined_parton_system, parton_ht
    
    def final_masking(self, data_array, combined_parton_system, parton_ht, num_phys_features=4):
        """
        Create padding mask for data_array and filter out fully padded events.
        Uses all `num_phys_features` channels to detect padding.
        Initially creates a boolean mask `pad_mask_np` where each entry is True if all the first
        `num_phys_features` features of a token are -99.0 (indicating padding).
        Then, it creates a `valid_mask` to identify events that are not fully padded (i.e., events
        that have at least one token with valid data).
        Finally, it filters the data_array, combined_parton_system, and pad_mask_np using the
        valid_mask to retain only the valid events.
        Args:
            data_array (numpy.ndarray): Input data array with shape (events, tokens, features).
            combined_parton_system (vector.Array): Combined parton system from the 4-momenta.
            num_phys_features (int): Number of physical features to consider for padding detection.
        
        Returns:
            tuple:
                X (numpy.ndarray): Filtered data array with shape (valid_events, tokens, features).
                y (numpy.ndarray): Corresponding target values with shape (valid_events,).
                pad_mask_np (numpy.ndarray): Padding mask for the filtered data array with shape (valid_events, tokens).
        """
        X = data_array.astype(np.float32)
        y1 = ak.to_numpy(combined_parton_system.mass).astype(np.float32)
        y2 = ak.to_numpy(parton_ht).astype(np.float32)
        y = np.stack((y1, y2), axis=1)  # Shape: (events, 2)
        pad_mask_np = (X[:, :, :4] == -99.0).all(axis=2)
        valid_mask = ~pad_mask_np.all(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        pad_mask_np = pad_mask_np[valid_mask]
        return X, y, pad_mask_np
    def __call__(self):
        """
        Main method to load and process data from the ROOT files.
        Also saves the files so that it's faster to run next time.
        Returns:
            tuple:
                X (numpy.ndarray): Final processed data array with shape (events, tokens, features).
                y (numpy.ndarray): Corresponding target values with shape (events,).
                pad_mask_np (numpy.ndarray): Padding mask for the data array with shape (events, tokens).
        """
        if os.path.exists("dual_large_cache_X.npy"):
            X = np.load("dual_large_cache_X.npy")
            y = np.load("dual_large_cache_y.npy")
            pad_mask_np = np.load("dual_large_cache_pad_mask.npy")
            return X, y, pad_mask_np
        data_array = []
        combined_parton_system = []
        parton_ht_list = []
        for filenames in self.files:
            file_data, parton_vectors, tree, parton_pt = self.data_loading(filenames)
            file_data, file_cps, file_parton_ht = self.initial_masking(file_data, parton_vectors, tree, parton_pt)
            file_data, file_cps, file_parton_ht = self.data_combination(file_data, file_cps, file_parton_ht)
            data_array.append(file_data)
            combined_parton_system.append(file_cps)
            parton_ht_list.append(file_parton_ht)
        data_array = np.concatenate(data_array, axis=0)
        combined_parton_system = ak.concatenate(combined_parton_system, axis=0)
        parton_ht = ak.concatenate(parton_ht_list, axis=0)
        X, y, pad_mask_np = self.final_masking(data_array, combined_parton_system, parton_ht, num_phys_features=4)

        # Randomise the data
        indices = np.arange(X.shape[0])
        np.random.seed(42)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        pad_mask_np = pad_mask_np[indices]

        np.save("dual_large_cache_X.npy", X)
        np.save("dual_large_cache_y.npy", y)
        np.save("dual_large_cache_pad_mask.npy", pad_mask_np)
        return X, y, pad_mask_np