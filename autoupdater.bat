if "%~1"==""  set update_text="automatic updates"
else set update_text=%1
call env export > environment.yml
git add *
git commit -m $update_text
git push