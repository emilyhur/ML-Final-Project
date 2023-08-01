#!/home/local/CORNELL/esh76/miniconda3/envs/btry4381/bin/python
print('Content-type: text/html')
print()

import sys
# Redirect error messages to stdout/screen (easy debugging)
sys.stderr = sys.stdout
# Import class to get parameters from HTML fields passed through GET/POST, etc
from cgi import FieldStorage
# Produeces nicely-formatted error messages
import cgitb
cgitb.enable()

print('''
<form action="./final_result.cgi" method="POST">
    <table border="2" width="700">
        <tr>
            <td align="left">
                Please input your <b>UniProt ID of interest</b>.
                (Max = 1)
            </td>
            <td align="left">
                <INPUT type="text" name="Protein">
            </td>
        </tr>
        <tr>
            <td align="center">
                <input type="submit" value="Submit">
            </td>
        </tr>
</form>
''')


##
#!/home/local/CORNELL/esh76/miniconda3/envs/btry4381/bin/python
print('Content-type: text/html')
print()

import sys
# Redirecty error messages to stdout/screen (easy debugging)
sys.stderr = sys.stdout
# Import class to get parameters from HTML fields passes through GET/POST, etc
from cgi import FieldStorage
# Produeces nicely-formatted error messages
import cgitb
cgitb.enable()

# Load and parse the data
from collections import defaultdict

inputfile = "../results.txt"
predictions = defaultdict(dict)
with open(inputfile, "r") as f:
    lines = f.readlines()
    for n in range(1, len(lines)):
        id, pos, AA, label = lines[n].strip().split("\t")
        predictions[id][pos]=label

# Get the input
form = FieldStorage()

query = form.getfirst("Protein", 0)

exist = False
rows = []
if query != 0:
    if query in predictions.keys():
        exist= True
        vals = predictions[query]
        for p in vals.keys():
            prediction = vals[p]
            temp = [p, prediction]
            rows.append(temp)


if exist:
    for q in rows:
        print('''
        <html>
            <body>
                <h3>The protein of interest: {0}</h3>
            </body>
        </html>
        ''')

else:
    print('''
    <html>
        <body>
            <center><img src="../not_found.png" width="600" height="400"></center>
        </body>
    </html>
    ''')