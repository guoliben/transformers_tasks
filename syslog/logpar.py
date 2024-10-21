from logparser.Drain import LogParser

input_dir = 'PATH_TO_LOGS/' # The input directory of log file
output_dir = 'result/'  # The output directory of parsing results
log_file = 'dlp.log'  # The input log file name
log_file = 'auth.log'  # The input log file name
log_format = '<Date> <Time> <Level>:<Content>' # Define log format to split message fields
# Regular expression list for optional preprocessing (default: [])
regex = [
    r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)' # IP
]
st = 0.5  # Similarity threshold
depth = 4  # Depth of all leaf nodes

parser = LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
parser.parse(log_file)
