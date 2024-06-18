#
# DEBUG Tool #1: read/write file detection
# USAGE: run first line before running 'python <script>.py' to detect what files are being read/written to
# for clarity, use second line to inspect only wrote-to files
# To ensure program is thread-safe, verify that write files are unique between runs (via a random identifier ideally)
#

TRACE_LOG_FILE=proc_read_write.txt

sudo fs_usage -w -e -f "filesys" cmd /opt/homebrew/bin/clingo | grep -E "/Users/tg4018/Tools/bin.*python|/Users/tg4018/Documents/.*" > $TRACE_LOG_FILE
cat $TRACE_LOG_FILE | grep "WrData" | grep -v $TRACE_LOG_FILE | awk '{print $(NF-3)}' | sort | uniq