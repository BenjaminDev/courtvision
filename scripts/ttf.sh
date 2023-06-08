jupyter nbconvert /Users/benjamindecharmoy/projects/courtvision/docs/presentation/TTF.ipynb --to slides
sed -i  's|slideNumber: "",|slideNumber: "", height:600, margin: 0.00,|'  /Users/benjamindecharmoy/projects/courtvision/docs/presentation/TTF.slides.html
# mv /Users/benjamindecharmoy/projects/courtvision/docs/presentation/TTF.slides.html /Users/benjamindecharmoy/projects/courtvision/docs/presentation/web/TTF.slides.html
python -m http.server -d .
