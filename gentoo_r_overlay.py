from os import path
import glob
import re
pkg_list = []
i = 0
for filename in glob.iglob('/r/a/p/var/lib/layman/R_Overlay/**/*.ebuild', recursive=True):
    name = path.basename(path.dirname(filename))
    print(name)
    with open(filename, "r") as f:
        text = f.read()
        #DESCRIPTION='HTML Widgets for R'
        lst = re.findall(r'''\n\s*DESCRIPTION\s*=\s*['"](.*)['"]''', text)
        if len(lst) > 0:
            pkg_list.append((name, lst[0]))
            i += 1
            # if i > 5:
            #     break
import pandas as pd
df = pd.DataFrame(pkg_list, columns=["name", "description"])
df.to_csv("gentoo_r_overlay.csv", encoding='utf-8')
