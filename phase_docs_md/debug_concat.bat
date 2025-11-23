@echo off
chcp 65001 > nul
echo Checking phase-1.md content:
type phase-1.md | findstr "열연"
echo.
echo Concatenating to phase-draft-new.md...
type phase-1.md > phase-draft-new.md
echo. >> phase-draft-new.md
echo. >> phase-draft-new.md
type phase-2.md >> phase-draft-new.md
echo. >> phase-draft-new.md
echo. >> phase-draft-new.md
type phase-3.md >> phase-draft-new.md
echo. >> phase-draft-new.md
echo. >> phase-draft-new.md
type phase-3.5.md >> phase-draft-new.md
echo. >> phase-draft-new.md
echo. >> phase-draft-new.md
type phase-4.md >> phase-draft-new.md
echo. >> phase-draft-new.md
echo. >> phase-draft-new.md
type phase-5.md >> phase-draft-new.md
echo. >> phase-draft-new.md
echo. >> phase-draft-new.md
type phase-6.md >> phase-draft-new.md
echo. >> phase-draft-new.md
echo. >> phase-draft-new.md
type phase-7.md >> phase-draft-new.md
echo. >> phase-draft-new.md
echo. >> phase-draft-new.md
type phase-8.md >> phase-draft-new.md
echo. >> phase-draft-new.md
echo. >> phase-draft-new.md
type phase-9.md >> phase-draft-new.md
echo. >> phase-draft-new.md
echo. >> phase-draft-new.md
type phase-10.md >> phase-draft-new.md
echo.
echo Checking phase-draft-new.md content:
type phase-draft-new.md | findstr "열연"
echo.
echo Replacing phase-draft.md...
del phase-draft.md
move phase-draft-new.md phase-draft.md
echo Done.
