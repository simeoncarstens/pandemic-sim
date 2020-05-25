{ pkgs ? import <nixpkgs> {} }:

with pkgs;
with pkgs.python36Packages;
let
  celluloid = buildPythonPackage rec {
    pname = "celluloid";
    version = "0.2.0";
    src = fetchPypi {
      inherit pname version;
      sha256 = "568b1512c4a97483759e9436c3f3e5dc5566da350179aa1872992ec8d82706e1";
    };
    propagatedBuildInputs = [ matplotlib ];
  };
in
  mkShell {
    buildInputs = [ celluloid numpy matplotlib ffmpeg ];
  }
