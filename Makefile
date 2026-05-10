ifeq ($(DEBIAN_BUILD),1)
PREFIX   ?= /usr/local
BINDIR    = $(DESTDIR)$(PREFIX)/bin
DATADIR   = $(DESTDIR)$(PREFIX)/share/i.hyper.cropname

PGM_PY   = i.hyper.cropname.py

$(PGM): $(PGM_PY)
	cp $< $@ && chmod 755 $@

install: $(PGM)
	install -d $(BINDIR)
	install -m 755 $(PGM) $(BINDIR)/i.hyper.cropname
	install -d $(DATADIR)
	install -m 644 data/spectral_library_ghisaconus_approx.csv $(DATADIR)/

clean:
	rm -f i.hyper.cropname

.PHONY: install clean

else
MODULE_TOPDIR = $(HOME)/dev/grass

PGM = i.hyper.cropname

include $(MODULE_TOPDIR)/include/Make/Script.make
include $(MODULE_TOPDIR)/include/Make/Html.make

default: script html $(TEST_DST)

$(HTMLDIR)/$(PGM).html: $(PGM).html
	$(INSTALL_DATA) $(PGM).html $(HTMLDIR)/$(PGM).html

endif
