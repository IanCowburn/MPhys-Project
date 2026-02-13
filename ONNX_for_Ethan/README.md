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

### 3. Pad the arrays and assemble the tensor

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

Then create the MET token at the end and add it to the input tensor, so that it should have final shape:

    (Events, 15, 6)

Then do the final masking and padding:

    X = data_array.astype(np.float32)
    y1 = ak.to_numpy(combined_parton_system.mass).astype(np.float32)
    y2 = ak.to_numpy(parton_ht).astype(np.float32)
    y = np.stack((y1, y2), axis=1)  # Shape: (events, 2)
    pad_mask_np = (X[:, :, :4] == -99.0).all(axis=2)
    valid_mask = ~pad_mask_np.all(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    pad_mask_np = pad_mask_np[valid_mask]

### 4. Scaling

This is the scaling for the invariant mass and HT:

    scaler_y = StandardScaler()
    y_train_flat = np.log(y_train).reshape(-1, 1)
    y_valid_flat = np.log(y_valid).reshape(-1, 1)
    y_test_flat  = np.log(y_test).reshape(-1, 1)
        
    y_train_scaled = scaler_y.fit_transform(y_train_flat).ravel()
    y_valid_scaled = scaler_y.transform(y_valid_flat).ravel()
    y_test_scaled = scaler_y.transform(y_test_flat).ravel()
    y_train_scaled = y_train_scaled.reshape(y_train.shape)
    y_valid_scaled = y_valid_scaled.reshape(y_valid.shape)
    y_test_scaled = y_test_scaled.reshape(y_test.shape)
    y_train_scaled = torch.from_numpy(y_train_scaled).float()
    y_valid_scaled = torch.from_numpy(y_valid_scaled).float()
    y_test_scaled = torch.from_numpy(y_test_scaled).float()

Clearly the datasets are split between train, valid and test.

The X values (reco observables) are only scaled for the first 4 features, i.e.

    X_for_scaling = X[:, :, :4]  # (E, T, 4) - kinematic features
    non_scaled_X = X[:, :, 4:]   # (E, T, 2) - charge, btag

The X values that are scaled are then scaled per feature across all events and tokens. 

### 5. Training

Finally, build the loaders and initialise the model:

    def build_loaders(self):
        train_ds = TensorDataset(self.X_train, self.y_train, self.mask_train)
        valid_ds = TensorDataset(self.X_valid, self.y_valid, self.mask_valid)
        test_ds  = TensorDataset(self.X_test,  self.y_test,  self.mask_test)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=self.batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=self.batch_size, shuffle=False)
        return train_loader, valid_loader, test_loader
    
    def init_model(self, device):
        model = TransformerRegressor(
            self.embed_dim,
            self.n_heads,
            self.num_layers,
            self.lepton_mask_size,
            self.jet_mask_size,
            self.in_features_leptons,
            self.in_features_jets
        ).to(device)
        return model

Use an early stopping of patience 5, adamW optimisation with:

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4 * (0.95 ** epoch), weight_decay=1e-4)

And use Huber loss as the loss function:

    loss_fn = nn.HuberLoss(delta = 100)

### 6. Hope it works!