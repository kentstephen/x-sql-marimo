import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell
def _(ctx):
    import pyarrow as pa
    from datafusion import udf
    from h3ronpy import cells_to_string
    from h3ronpy.arrow.vector import coordinates_to_cells  # check the exact module path for your version

    def latlng_to_cell(lat, lng, res):
        return pa.array(coordinates_to_cells(lat.to_numpy(), lng.to_numpy(), res[0].as_py()))

    h3_cell = udf(latlng_to_cell, [pa.float64(), pa.float64(), pa.int32()],
                  pa.uint64(), "stable", name="h3_latlng_to_cell")
    ctx.register_udf(h3_cell)
    return


if __name__ == "__main__":
    app.run()
