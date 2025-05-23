#!/bin/sh
skip=49

tab='	'
nl='
'
IFS=" $tab$nl"

umask=`umask`
umask 77

gztmpdir=
trap 'res=$?
  test -n "$gztmpdir" && rm -fr "$gztmpdir"
  (exit $res); exit $res
' 0 1 2 3 5 10 13 15

case $TMPDIR in
  / | /*/) ;;
  /*) TMPDIR=$TMPDIR/;;
  *) TMPDIR=/tmp/;;
esac
if type mktemp >/dev/null 2>&1; then
  gztmpdir=`mktemp -d "${TMPDIR}gztmpXXXXXXXXX"`
else
  gztmpdir=${TMPDIR}gztmp$$; mkdir $gztmpdir
fi || { (exit 127); exit 127; }

gztmp=$gztmpdir/$0
case $0 in
-* | */*'
') mkdir -p "$gztmp" && rm -r "$gztmp";;
*/*) gztmp=$gztmpdir/`basename "$0"`;;
esac || { (exit 127); exit 127; }

case `printf 'X\n' | tail -n +1 2>/dev/null` in
X) tail_n=-n;;
*) tail_n=;;
esac
if tail $tail_n +$skip <"$0" | gzip -cd > "$gztmp"; then
  umask $umask
  chmod 700 "$gztmp"
  (sleep 5; rm -fr "$gztmpdir") 2>/dev/null &
  "$gztmp" ${1+"$@"}; res=$?
else
  printf >&2 '%s\n' "Cannot decompress $0"
  (exit 127); res=127
fi; exit $res
��r�gtest.sh �T[OA~�_q�n	M�^H�SL�����f۝v�lw��TLj�"�`xP0B$����T"���ɿ���.Tna�ݙ�|�;�93��n�s���L8J������ɑ�L�Kg��0	Rd�� L]�1� >,�Y@[����n������uIR�	I�����2:t-E�j�Fwoiv����^]�ll4�7�?�L��F�S����~�}s���<v?<i��i��lm��/ʹ�δ��p�<m/|�!�p�l ��@�7��ݵ���F��7w���ީ��"���=4~#E�e�V�Ym:��f\���Z��;Ù��� �Y%�{#隇�]��䥧 у��r7�:�o��]{����c��%�HX�I\�zd���(F�PRt3�σly� � �Uff��m,s�f�
���T>M&	�g�t5%��w*9��i��zՠ�bg#B�"2�\I眩��0�sF҃ y<(��%fUx�r"� �P�SS	����#u��iM7:v�*���OL_���A��-�U-�.~��2HA��nr��?`�B��D�j��ۻ3���xWz�טz�?>Zǡ�Lf+hO�XWq��~��=G�}��^��,?s��n�6ڋ���?!.qr���}����������O���B�P�Pf8�j:�AAt���bF��öm٧F��H���G��X\u0�R5����Y���+f^�"{@х(
��+��	�D��|Ŷ�ɳb��b4Di��.�aUߐ�7�p=!�Ů"�Y	.��w-d��#�J�=�#
A��7��HGU�n��|zc(��իH���U��	  