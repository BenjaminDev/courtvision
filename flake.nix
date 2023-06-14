{
  description = "Example Python development environment for Zero to Nix";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  };


  # Flake outputs
  outputs = { self, nixpkgs }:

    let
      # Systems supported
      allSystems = [
        "x86_64-linux" # 64-bit Intel/AMD Linux
        "aarch64-linux" # 64-bit ARM Linux
        "x86_64-darwin" # 64-bit Intel macOS
        "aarch64-darwin" # 64-bit ARM macOS
      ];

      # Helper to provide system-specific attributes
      forAllSystems = f: nixpkgs.lib.genAttrs allSystems (system: f {
        pkgs = import nixpkgs { inherit system; };
      });
      pkgs = import nixpkgs { };
    in
    {
      # Development environment output
      devShells = forAllSystems ({ pkgs }: {
        default =
          let
            # Use Python 3.11
            python = pkgs.python311;
          in
          pkgs.mkShell {
            # The Nix packages provided in the environment
            packages = [
              pkgs.ffmpeg
              pkgs.git-lfs
              pkgs.poetry
              pkgs.ruff
              pkgs.zsh
              pkgs.starship
              pkgs.podman
              pkgs.qemu
              pkgs.docker-compose
              # Python plus helper tools
              (python.withPackages (ps: with ps; [
                virtualenv # Virtualenv
                pip # The pip installer
              ]))
            ];
            # Environment variables to set
            shellHook = ''
              eval "$(starship init zsh)"
              poetry config virtualenvs.create = true --local
              poetry virtualenvs.in-project = true --local
              poetry install
            '';
          };

      });
      # ffmpeg = pkgs.ffmpeg;

    };
}
