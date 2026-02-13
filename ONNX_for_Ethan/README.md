# Hi Ethan 🐔

## Transformer Dataloading information:

### 1. Loading the files

Files used:

files = ["tttt_NLO_523243_mc23a_fullsim.root",
         "tttt_NLO_523243_mc23d_fullsim.root",
         "tttt_NLO_523243_mc23e_fullsim.root"]

Only consider events with up to two (charged) leptons and up to 12 jets.

#### This means you get 15 tokens:

##### 2 Lepton tokens:

(Eta, Phi, Pt, E, Lepton charge, B-jet tagging (i.e. 0))

##### 12 Jet tokens:

(Eta, Phi, Pt, E, Lepton charge (i.e. 0), B-jet tagging)

##### 1 MET token:

(Eta (i.e. 0), MET Phi, MET MET, E (i.e. 0), Lepton charge (i.e. 0), B-jet tagging (i.e. 0))