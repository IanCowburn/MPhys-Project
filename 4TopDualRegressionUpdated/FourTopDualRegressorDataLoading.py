
# Imports
import uproot
import awkward as ak
import vector
import numpy as np
import os
# Data files
files = ["scratch/tttt_NLO_523243_mc23a_fullsim.root",
         "scratch/tttt_NLO_523243_mc23d_fullsim.root",
         "scratch/tttt_NLO_523243_mc23e_fullsim.root"]
var_names = ["lepton_eta", "lepton_phi", "jet_eta", "jet_phi", "lepton_pt_NOSYS", "jet_pt_NOSYS", "lepton_e_NOSYS", "jet_e_NOSYS", "met_met_NOSYS", "met_phi_NOSYS", "jet_GN2v01_FixedCutBEff_77_select", "lepton_charge"]
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
        tree = file_tt["reco"]
        parton_vectors = vector.zip({
            "pt": tree["parton_top_pt"].array(),
            "eta": tree["parton_top_eta"].array(),
            "phi": tree["parton_top_phi"].array(),
            "mass": tree["parton_top_m"].array()
        })
        parton_pt = tree["parton_top_pt"].array()
        data_array = tree.arrays(self.var_names)
        
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
            
        padding_size = []
        masking_array = tree.arrays(["jet_eta", "lepton_eta"])
        nLeptons_mask = ak.num(masking_array["lepton_eta"]) <= self.lepton_mask_size
        nJets_mask = ak.num(masking_array["jet_eta"]) <= self.jet_mask_size
        data_array = data_array[nLeptons_mask & nJets_mask]
        parton_vectors = parton_vectors[nLeptons_mask & nJets_mask]
        combined_parton_system = parton_vectors[:,0] + parton_vectors[:,1] + parton_vectors[:,2] + parton_vectors[:,3]
        parton_pt = parton_pt[nLeptons_mask & nJets_mask]
        parton_ht = ak.sum(parton_pt, axis=1)
        for name in self.var_names:
            # Skip MET variables - they are event-level scalars, not token arrays
            if "met" in name:
                continue
            pad = int(ak.max(ak.num(data_array[name])))
            padding_size.append(pad)
        for i in range(len(self.var_names)):
            name = self.var_names[i]
            # Skip MET variables
            if "met" in name:
                continue
            pad = padding_size[i - sum(1 for v in self.var_names[:i] if "met" in v)]
            try:
                data_array[name] = ak.pad_none(data_array[name], pad, clip=True)
                data_array[name] = ak.fill_none(data_array[name], -99)
            except Exception:
                pass
        
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
        for name in self.var_names:
            if "lepton" in name:
                lepton_arrays.append(data_array[name])
            elif "jet" in name:
                jet_arrays.append(data_array[name])
            elif "met" in name:
                met_arrays.append(data_array[name])
        
        lepton_arrays = ak.to_numpy(lepton_arrays)  # (5, events, lepton_tokens)
        jet_arrays = ak.to_numpy(jet_arrays)        # (5, events, jet_tokens)
        met_arrays = ak.to_numpy(met_arrays)        # (2, events) - event-level scalars
        
        num_events = lepton_arrays.shape[1]
        
        # Concatenate eta, phi, pt, e along token axis: (4, events, total_tokens)
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
        
        # Create MET token: shape (events, 1, 6) with [0, met_phi, met_met, 0, 0, 0]
        met_token = np.zeros((num_events, 1, 6), dtype=np.float32)
        met_token[:, 0, 1] = met_arrays[1, :]  # phi = met_phi
        met_token[:, 0, 2] = met_arrays[0, :]  # pt = met_met
        
        # Stack all features for leptons+jets: (6, events, 14)
        data_array = np.concatenate([
            data_array,                            # eta, phi, pt, e (4 features)
            charge_leptons[np.newaxis, :, :],     # charge (1 feature)
            bjet_tags[np.newaxis, :, :]           # b-jet tag (1 feature)
        ], axis=0)
        
        # Transpose to (events, 14, 6)
        data_array = np.transpose(data_array, (1, 2, 0))
        
        # Append MET token as 15th token: (events, 15, 6)
        data_array = np.concatenate([data_array, met_token], axis=1)
        
        return data_array, combined_parton_system, parton_ht
    
    def lepton_jet_classifier(self, x):
        """
        Creates a classifier array to distinguish between leptons and jets in the data array.
        Args:
            x (numpy.ndarray): Input data array with shape (events, tokens, features).
        
        Returns:
            classifier_array (numpy.ndarray): Classifier array with shape (events, tokens),
            where leptons are labeled as 1 and jets as 2.
        """
        events_length = x.shape[0]
        lepton_classifier_array = np.ones((events_length, self.lepton_mask_size), dtype=np.float32)
        jet_classifier_array = 2 * np.ones((events_length, self.jet_mask_size), dtype=np.float32)
        classifer_array = np.concatenate((lepton_classifier_array, jet_classifier_array), axis=1)
        return classifer_array
    
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
        if os.path.exists("dual_cache_X.npy"):
            X = np.load("dual_cache_X.npy")
            y = np.load("dual_cache_y.npy")
            pad_mask_np = np.load("dual_cache_pad_mask.npy")
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
        # classifier_array = self.lepton_jet_classifier(data_array)
        # base_pad_mask = (data_array == -99.0).all(axis=2)
        # classifier_array[base_pad_mask] = -99.0
        # data_array = np.concatenate((data_array, classifier_array[:, :, np.newaxis].astype(np.float32)), axis=2)
        X, y, pad_mask_np = self.final_masking(data_array, combined_parton_system, parton_ht, num_phys_features=4)
        np.save("dual_cache_X.npy", X)
        np.save("dual_cache_y.npy", y)
        np.save("dual_cache_pad_mask.npy", pad_mask_np)
        return X, y, pad_mask_np