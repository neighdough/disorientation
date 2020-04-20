#!/usr/sh
dissertation_dir="$HOME/Dropbox/phd/dissertation"
cd ${dissertation_dir}
office_dir="$HOME/Dropbox/phd/dissertation/office"
inotifywait -r -m -q -e  modify --format '%w%f' ${dissertation_dir} | 
	while read FILE; do
	if [[ "$FILE" =~ .*ods$ ]]; then
		echo "Saving ${FILE} to ${office_dir}"
		soffice --convert-to xlsx --outdir ${office_dir} *.ods --headless
	fi
	if [[ "${FILE}" =~ .*odt$ ]]; then
		echo "Converting to Word"
		soffice --convert-to docx --outdir ${office_dir} *.odt --headless
	fi
done

