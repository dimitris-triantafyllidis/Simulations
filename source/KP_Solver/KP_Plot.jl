using CSV
using DataFrames
using CairoMakie

folder = "/home/dimitris/Documents/KP_Snapshots"

files = sort(
    filter(f -> endswith(f, ".txt"), readdir(folder)),
    by = f -> parse(Int, match(r"\d+", f).match)
)

lx = 20.0
ly = 20.0

for file in files

    println("Plotting $file")

    filename = joinpath(folder, file)

    # Read data
    u = Matrix(CSV.read(filename, DataFrame; strict=true, header=false))

    nx, ny = size(u)

    x = collect(range(-lx/2, lx/2, length=nx))
    y = collect(range(-ly/2, ly/2, length=ny))

    fig = Figure(size = (1000, 1000))

    ax = Axis3(
        fig[1, 1],
        xlabel = "y",
        ylabel = "x",
        zlabel = "u",
        title = file
    )

    surface!(ax, x, y, u, colormap = :magma)

    zlims!(ax, (0, 10))

    ax.azimuth = 45 * pi / 180
    ax.elevation = 30 * pi / 180

    outname = replace(file, ".txt" => ".png")
    save(joinpath(folder, outname), fig, px_per_unit = 1)

end

println("Finished plotting $(length(files)) files.")
