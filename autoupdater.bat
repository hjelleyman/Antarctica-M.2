if "%~1"==""  "automatic updates" => update_text
else %1 => update_text
call env export > environment.yml
git add *
git commit -m update_text
git push