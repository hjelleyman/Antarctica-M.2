if "%~1"==""  SET update_text="automatic updates"
else SET update_text=%1
call env export > environment.yml
git add *
git commit -m $update_text
git push