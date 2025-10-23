import urllib.request
urllib.request.urlretrieve("https://cernbox.cern.ch/s/nrBbuO7bu4wi82W/download", "ttbar_from_cernbox.root")

import uproot
file_tt = uproot.open("ttbar_from_cernbox.root")
file_tt.keys()