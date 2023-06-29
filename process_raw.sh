raw_dir=data/raw
for raw_filename in "$raw_dir"/*; do
  fits_filename="${raw_filename/.NEF/.FITS}"
  fits_filename="${fits_filename/.CR2/.FITS}"
  fits_filename="${fits_filename/raw/fits}"
  dcraw -c -w -v -4 -S 65535 $raw_filename | pnmtofits >$fits_filename #
done
