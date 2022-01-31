from functools import wraps

from flask import abort, Blueprint, jsonify, request, send_from_directory, flash,redirect,url_for
from flask_login import current_user, login_user
from passlib.hash import sha256_crypt
from flask_login import login_required, current_user, login_user, logout_user

from app import data_loader, date_time_transformer, data_transformer, numerical_transformer, one_hot_encoder, \
    data_deduplicator, active_user_handler, UPLOAD_FOLDER,data_classifier,data_regression
from app.history.models import History
from app.user_service.models import UserDataAccess

from flask import  render_template, make_response

api = Blueprint('api', __name__)
 

_history = History()
 

def auth_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth = request.authorization
        if not auth:
            if current_user.is_authenticated:
                return f(*args, **kwargs)
            else:
                abort(401)
        user = UserDataAccess().get_user(user_id=auth.username)
        retrieved_pass = UserDataAccess().login_user(auth.username)
        if user is None or sha256_crypt.verify(auth.password, retrieved_pass):
            abort(401)
        login_user(user)
        return f(*args, **kwargs)

    return wrapper
 

@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>', methods=['GET'])
@auth_required
def get_table(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
    start = request.args.get('start')
    length = request.args.get('length')
    order_column_idx = int(request.args.get('order[0][column]'))
    order_column_name = request.args.get('columns[{}][data]'.format(order_column_idx))
    order_direction = request.args.get('order[0][dir]')
    ordering = (order_column_name, order_direction)
    search = request.args.get('search[value]')

    table = data_loader.get_table(dataset_id, table_name, offset=start, limit=length, ordering=ordering, search=search)
    # Make proper data dict out of table rows
    data = list()
    for r_ix in range(len(table.rows)):
        r = dict()
        for c_ix in range(len(table.columns)):
            r[table.columns[c_ix].name] = table.rows[r_ix][c_ix]
        data.append(r)

    return jsonify(draw=int(request.args.get('draw')),
                   recordsTotal=table.total_size,
                   recordsFiltered=table.total_size,
                   data=data)  # table.rows


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/history', methods=['GET'])
@auth_required
def get_history(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
    start = request.args.get('start')
    length = request.args.get('length')
    search = request.args.get('search[value]')
    order_column = int(request.args.get('order[0][column]'))
    order_direction = request.args.get('order[0][dir]')
    ordering = (['date', 'action_desc'][order_column], order_direction)

    rows = _history.get_actions(dataset_id, table_name, offset=start, limit=length, ordering=ordering, search=search)
    _rows = _history.get_actions(dataset_id, table_name)

    return jsonify(draw=int(request.args.get('draw')),
                   recordsTotal=len(_rows),
                   recordsFiltered=len(_rows),
                   data=rows)


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/history/undo/<int:action_id>', methods=['POST'])
@auth_required
def undo_action(dataset_id, table_name, action_id):
    if not data_loader.has_access(current_user.username, dataset_id):
        return abort(403)
    active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
    try:
        _history.undo_action(dataset_id, table_name, action_id)
        flash(u"Action was undone", 'success')
        return jsonify({'success': True}), 200
    except Exception as e:
        flash(u"Action could not be undone", 'danger')
        return jsonify({'error': True}), 400


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/rows', methods=['POST'])
@auth_required
def add_row(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        values = dict()
        columns = list()
        for key in request.args:
            if key.startswith('value-col'):
                col_name = key.split('-')[2]  # Key is of the form "value-col-[name]"
                values[col_name] = request.args.get(key)
                columns.append(col_name)
        data_loader.insert_row(table_name, dataset_id, columns, values)
        flash(u"Rows have been added.", 'success')
        return jsonify({'success': True}), 200
    except Exception:
        flash(u"Rows couldn't be added.", 'danger')
        return jsonify({'error': True}), 400


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/rows', methods=['DELETE'])
@auth_required
def delete_row(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        row_ids = [key.split('-')[1] for key in request.args]
        data_loader.delete_row(dataset_id, table_name, row_ids)
        flash(u"Rows have been deleted.", 'success')
        return jsonify({'success': True}), 200
    except Exception:
        flash(u"Rows couldn't be deleted.", 'danger')
        return jsonify({'error': True}), 400

@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/edit_row', methods=['POST'])
@auth_required
def edit_row(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        row_ids = [key.split('-')[1] for key in request.args]
        print(row_ids)
        table=data_loader.get_table_by_id(dataset_id, table_name, row_ids[0])
        print(table)

        data = list()
        for r_ix in range(len(table.rows)):
            r = dict()
            for c_ix in range(len(table.columns)):
                r[table.columns[c_ix].name] = table.rows[r_ix][c_ix]
            data.append(r)
        print(data)

        #return render_template('data_service/modals/modal-edit-row.html')

        flash(u"Rows have been deeeeleted.", 'success')
         

         
        return render_template('data_service/modals/modal-edit-row.html')
        
        '''return jsonify(draw=int(request.args.get('draw')),
                   recordsTotal=table.total_size,
                   recordsFiltered=table.total_size,
               data=data)  # table.rows
        '''
        
        #return jsonify({'success': True}), 200
        
    except Exception:
        print(Exception)
        flash(u"Rows couldn't be deleted.", 'danger')
        return jsonify({'error': True}), 400



@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/columns', methods=['POST'])
@auth_required
def add_column(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        column_name = request.args.get('col-name')
        column_type = request.args.get('col-type')
        data_loader.insert_column(dataset_id, table_name, column_name, column_type)
        flash(u"Column has been added.", 'success')
        return jsonify({'success': True}), 200
    except Exception:
        flash(u"Column couldn't be added.", 'danger')
        return jsonify({'error': True}), 400


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/columns', methods=['PUT'])
@auth_required
def update_column(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        column_name = request.args.get('col-name')
        column_type = request.args.get('col-type')
        data_loader.update_column_type(dataset_id, table_name, column_name, column_type)
        flash(u"Column type has been updated.", 'success')
        return jsonify({'success': True}), 200
    except Exception:
        flash(u"Column type couldn't be updated.", 'danger')
        return jsonify({'error': True}), 400


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/columns', methods=['DELETE'])
@auth_required
def delete_column(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        column_name = request.args.get('col-name')
        data_loader.delete_column(dataset_id, table_name, column_name)
        flash(u"Column has been deleted.", 'success')
        return jsonify({'success': True}), 200
    except Exception:
        flash(u"Column couldn't be deleted.", 'danger')
        return jsonify({'error': True}), 400


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/date-time-transformations', methods=['PUT'])
@auth_required
def transform_date_or_time(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        column_name = request.args.get('col-name')
        operation_name = request.args.get('operation-name')
        date_time_transformer.transform(dataset_id, table_name, column_name, operation_name)
        flash(u"Date/Time transformation was successful.", 'success')
        return jsonify({'success': True}), 200
    except Exception:
        flash(u"Date/Time transformation was unsuccessful.", 'danger')
        return jsonify({'error': True}), 400


@api.route('/api/datasets/update-dataset-metadata', methods=['PUT'])
@auth_required
def update_dataset_metadata():
    try:
        dataset_id = request.args.get('ds-id')
        new_name = request.args.get('ds-name')
        new_desc = request.args.get('ds-desc')
        if (data_loader.has_access(current_user.username, dataset_id)) is False:
            return abort(403)
        data_loader.update_dataset_metadata(dataset_id, new_name, new_desc)
        flash(u"Metadata has been updated.", 'success')
        return jsonify({'success': True}), 200
    except Exception:
        flash(u"Metadata couldn't be updated.", 'danger')
        return jsonify({'error': True}), 400


@api.route('/api/datasets/<int:dataset_id>/update-metadata', methods=['PUT'])
@auth_required
def update_table_metadata(dataset_id):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_dataset(dataset_id, current_user.username)
        old_table_name = request.args.get('t-old-name')
        new_table_name = request.args.get('t-name').replace('"', '')
        if not len(new_table_name):
            raise Exception('Could not rename table')
        new_desc = request.args.get('t-desc')
        data_loader.update_table_metadata(dataset_id, old_table_name, new_table_name, new_desc)
        flash(u"Metadata has been updated.", 'success')
        return jsonify({'success': True}), 200
    except Exception:
        flash(u"Metadata couldn't be updated.", 'danger')
        return jsonify({'error': True}), 400


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/impute-missing-data', methods=['PUT'])
@auth_required
def impute_missing_data(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        column_name = request.args.get('col-name')
        function = request.args.get('function')
        if function == "CUSTOM":
            custom_value = request.args.get('custom-value')
            data_transformer.impute_missing_data(dataset_id, table_name, column_name, function, custom_value)
        else:
            data_transformer.impute_missing_data(dataset_id, table_name, column_name, function)
        flash(u"Missing data has been filled.", 'success')
        return jsonify({'success': True}), 200
    except Exception:
        flash(u"Couldn't fill missing data.", 'danger')
        return jsonify({'error': True}), 400


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/export', methods=['PUT'])
@auth_required
def export_table(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    # Maybe later we might add other types, but for now this is hardcoded to export as CSV
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        filename = table_name + ".csv"
        path = UPLOAD_FOLDER + "/" + filename

        separator = request.args.get('separator')
        quote_char = request.args.get('quote_char')
        empty_char = request.args.get('empty_char')

        data_loader.export_table(path, dataset_id, table_name, separator=separator, quote_char=quote_char,
                                 empty_char=empty_char)
        flash(u"Data has been exported.", 'success')
        return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)
    except Exception:
        flash(u"Data couldn't be exported.", 'danger')
        return jsonify({'error': True}), 400


@api.route('/api/download/<string:filename>', methods=['GET'])
@auth_required
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/show-raw-data', methods=['GET'])
@auth_required
def show_raw_data(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
    start = request.args.get('start')
    length = request.args.get('length')
    order_column = int(request.args.get('order[0][column]'))
    order_direction = request.args.get('order[0][dir]')
    raw_table_name = "_raw_" + table_name
    ordering = (data_loader.get_column_names(dataset_id, raw_table_name)[order_column], order_direction)
    table = data_loader.get_table(dataset_id, raw_table_name, offset=start, limit=length, ordering=ordering)
    _table = data_loader.get_table(dataset_id, raw_table_name)
    return jsonify(draw=int(request.args.get('draw')),
                   recordsTotal=len(_table.rows),
                   recordsFiltered=len(_table.rows),
                   data=table.rows)


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/find-and-replace', methods=['PUT'])
@auth_required
def find_and_replace(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        colomn = request.args.get('col-name')
        replacement_function = request.args.get('replacement-function')

        replacement_value = request.args.get('replacement-value')
        if replacement_function == "regex":
            regex = request.args.get('replacement-regex')
            data_transformer.find_and_replace_by_regex(dataset_id, table_name, colomn, regex, replacement_value)
        else:

            value_to_be_replaced = request.args.get('value-to-be-replaced')
            data_transformer.find_and_replace(dataset_id, table_name, colomn, value_to_be_replaced, replacement_value,
                                              replacement_function)
        flash(u"Find and replace was successful.", 'success')
        return jsonify({'success': True}), 200
    except Exception:
        flash(u"Find and replace was unsuccessful.", 'danger')
        return jsonify({'error': True}), 400


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/normalize', methods=['PUT'])
@auth_required
def normalize(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        column_name = request.args.get('col-name')
        numerical_transformer.normalize(dataset_id, table_name, column_name)
        flash(u"Data has been normalized.", 'success')
        return jsonify({'success': True}), 200
    except Exception:
        flash(u"Data couldn't be normalized.", 'danger')
        return jsonify({'error': True}), 400


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/discretize', methods=['PUT'])
@auth_required
def discretize(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    column_name = request.args.get('col-name')
    discretization = request.args.get('discretization')
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        if discretization == 'eq-width':
            num_intervals = int(request.args.get('num-intervals'))
            numerical_transformer.equal_width_interval(dataset_id, table_name, column_name, num_intervals)
        elif discretization == 'eq-freq':
            num_intervals = int(request.args.get('num-intervals'))
            numerical_transformer.equal_freq_interval(dataset_id, table_name, column_name, num_intervals)
        elif discretization == 'manual':
            intervals = [int(n) for n in request.args.get('intervals').strip().split(',')]
            numerical_transformer.manual_interval(dataset_id, table_name, column_name, intervals)
        else:
            flash(u"Data couldn't be discritized.", 'danger')
            return jsonify({'error': True}), 400
    except ValueError:
        flash(u"Data couldn't be discritized.", 'danger')
        return jsonify({'error': True}), 400
    flash(u"Data has been discritized.", 'success')
    return jsonify({'success': True}), 200


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/outliers', methods=['PUT'])
@auth_required
def outliers(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        column_name = request.args.get('col-name')
        option = request.args.get('option') == 'less-than'
        value = float(request.args.get('value'))
        numerical_transformer.remove_outlier(dataset_id, table_name, column_name, value, option)
    except ValueError:
        flash(u"Outliers couldn't be removed.", 'danger')
        return jsonify({'error': True}), 400
    flash(u"Outliers have been removed.", 'success')
    return jsonify({'success': True}), 200


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/rename-column', methods=['PUT'])
@auth_required
def rename_column(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        to_rename = request.args.get('col-name')
        new_name = request.args.get('new-name').replace('"', '')
        if not len(new_name) or new_name == to_rename:
            raise Exception('Column name is empty')
        data_loader.rename_column(dataset_id, table_name, to_rename, new_name)
        flash(u"Column has been renamed.", 'success')
        return jsonify({'success': True}), 200
    except Exception:
        flash(u"Column couldn't be renamed.", 'danger')
        return jsonify({'error': True}), 400


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/chart', methods=['GET'])
@auth_required
def chart(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        column_name = request.args.get('col-name')
        column_type = request.args.get('col-type')

        if column_type not in ['real', 'double', 'integer', 'timestamp']:
            return jsonify(numerical_transformer.chart_data_categorical(dataset_id, table_name, column_name))
        else:
            return jsonify(numerical_transformer.chart_data_numerical(dataset_id, table_name, column_name))
    except Exception:
        flash(u"Charts couldn't be produced.", 'danger')
        return jsonify({'error': True}), 400


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/one-hot-encode-column', methods=['PUT'])
@auth_required
def one_hot_encode(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        column_name = request.args.get('col-name')
        one_hot_encoder.encode(dataset_id, table_name, column_name)
        flash(u"One hot encoding was successful.", 'success')
        return jsonify({'success': True}), 200
    except Exception:
        flash(u"One hot encoding was unsuccessful.", 'danger')
        return jsonify({'error': True}), 400


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/remove-identical-rows-alg', methods=['PUT'])
@auth_required
def collect_identical_rows_alg(dataset_id, table_name):
    try:
        sorting_key = request.args.get('selected-col-name')
        fixed_column_names = request.args.getlist('identical-col-names')
        var_column_names = request.args.getlist('similar-col-names')
        duplicates_found = data_deduplicator.collect_identical_rows_alg(dataset_id, table_name, sorting_key,
                                                                        fixed_column_names, var_column_names,
                                                                        'damerau_levenshtein')

        if not duplicates_found:
            flash(u"No identical rows could be found", 'danger')
            return jsonify({'error': True}), 400

        return jsonify({'success': True}), 200
    except Exception:
        flash(u"Couldn't collect identical rows", 'danger')
        return jsonify({'error': True}), 400


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/show-dedup-data-alg', methods=['GET'])
@auth_required
def get_duplicate_group(dataset_id, table_name):
    try:
        start = request.args.get('start')
        length = request.args.get('length')
        order_column = int(request.args.get('order[0][column]'))
        order_direction = request.args.get('order[0][dir]')
        ordering = (data_loader.get_column_names(dataset_id, table_name)[order_column], order_direction)
        search = request.args.get('search[value]')

        group_id = data_deduplicator.get_next_group_id(dataset_id, table_name)

        table = data_deduplicator.get_cluster(dataset_id, table_name, group_id=group_id, offset=start, limit=length,
                                              ordering=ordering,
                                              search=search)
        _table = data_deduplicator.get_cluster(dataset_id, table_name, group_id=group_id)

        return jsonify(draw=int(request.args.get('draw')),
                       recordsTotal=len(_table.rows),
                       recordsFiltered=len(_table.rows),
                       data=table.rows)
    except Exception:
        flash(u"Cluster of duplicate rows could't be shown.", 'danger')
        return jsonify({'error': True}), 400


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/active-users', methods=['GET'])
@auth_required
def get_active_users(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        active_users = active_user_handler.get_active_users_in_table(dataset_id, table_name, current_user.username)
        return jsonify(data=active_users)
    except Exception:
        return jsonify({'error': True}), 400


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/create-backup', methods=['PUT'])
@auth_required
def create_backup(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        if not data_loader.backup_available(dataset_id, table_name):
            flash(u"You have reached the limit amount of backups. You must remove a backup to create a new one.",
                  'danger')
            return jsonify({'error': True}), 400
        note = request.args.get('backup-note')
        data_loader.make_backup(dataset_id, table_name, note)
        flash(u"Succesfully created backup.", 'success')
        return jsonify({'success': True}), 200
    except Exception as e:
        if e.__str__() == "Backup limit reached.":
            flash(u"Can't create backup, limit reached.", 'danger')
        flash(u"Failed to create backup.", 'danger')
        return jsonify({'error': True}), 400


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/restore-backup', methods=['GET'])
@auth_required
def restore_backup(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        backup_ts = request.args.get('backup-timestamp')
        if backup_ts == "DEFAULT":
            return jsonify({'error': True}), 400
        data_loader.restore_backup(dataset_id, table_name, backup_ts)
        flash(u"Succesfully restored backup.", 'success')
        return jsonify({'success': True}), 200
    except Exception:
        flash(u"Failed to restore backup.", 'danger')
        return jsonify({'error': True}), 400


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/delete-backup/<string:timestamp>',
           methods=['DELETE'])
@auth_required
def delete_backup(dataset_id, table_name, timestamp):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        data_loader.delete_backup(dataset_id, table_name, timestamp)
        return jsonify({'success': True}), 200
    except Exception:
        return jsonify({'error': True}), 400


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/get-backup-info/<string:timestamp>',
           methods=['GET'])
@auth_required
def get_backup_info(dataset_id, table_name, timestamp):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        if timestamp == "DEFAULT":
            return "Select backup to display note..."
        note = data_loader.get_backup_info(dataset_id, table_name, timestamp)
        print(note)
        return note
    except Exception:
        return ""


@api.route('/api/admin-page/add-admin', methods=['POST'])
@auth_required
def add_admin():
    to_add = request.args.get('ap-add-admin-name')
    try:
        UserDataAccess().set_admin(to_add)
        flash(u"Admin rights granted.", 'success')
        return jsonify({'success': True}), 200
    except Exception as e:
        flash(u"Failed to grant admin rights.", 'danger')
        return jsonify({'error': True}), 400


@api.route('/api/admin-page/remove-admin', methods=['POST'])
@auth_required
def remove_admin():
    to_remove = request.args.get('ap-remove-admin-select')
    try:
        UserDataAccess().set_admin(to_remove, False)
        flash(u"Admin rights removed.", 'success')
        return jsonify({'success': True}), 200
    except Exception as e:
        flash(u"Failed to remove admin rights.", 'danger')
        return jsonify({'error': True}), 400






@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/regressionLinear', methods=['PUT'])
@auth_required
def regressionLinear(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        column_name = request.args.get('col-name')
        test_ratio = request.args.get('test-ratio')
        message=data_regression.linear(dataset_id, table_name, column_name,test_ratio)
        flash(u"\n\n"+message, 'success')
        return jsonify({'success': True}), 200
    except Exception:
        flash(u"Linear Regression can not be applied"+str(test_ratio), 'danger')
        return jsonify({'error': True}), 400

@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/regressionElastic', methods=['PUT'])
@auth_required
def regressionElastic(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        column_name = request.args.get('col-name')
        test_ratio = request.args.get('test-ratio')
        message=data_regression.elastic(dataset_id, table_name, column_name,test_ratio)
        flash(u"\n\n"+message, 'success')
        return jsonify({'success': True}), 200
    except Exception:
        flash(u"Elastic Regression can not be applied"+str(test_ratio), 'danger')
        return jsonify({'error': True}), 400

@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/regressionLasso', methods=['PUT'])
@auth_required
def regressionLasso(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        column_name = request.args.get('col-name')
        test_ratio = request.args.get('test-ratio')
        message=data_regression.lasso(dataset_id, table_name, column_name,test_ratio)
        flash(u"\n\n"+message, 'success')
        return jsonify({'success': True}), 200
    except Exception:
        flash(u"Lasso Regression can not be applied"+str(test_ratio), 'danger')
        return jsonify({'error': True}), 400


@api.route('/api/datasets/<int:dataset_id>/tables/<string:table_name>/regressionSVR', methods=['PUT'])
@auth_required
def regressionSVR(dataset_id, table_name):
    if (data_loader.has_access(current_user.username, dataset_id)) is False:
        return abort(403)
    try:
        active_user_handler.make_user_active_in_table(dataset_id, table_name, current_user.username)
        column_name = request.args.get('col-name')
        test_ratio = request.args.get('test-ratio')
        message=data_regression.svr(dataset_id, table_name, column_name,test_ratio)
        flash(u"\n\n"+message, 'success')
        return jsonify({'success': True}), 200
    except Exception:
        flash(u"SVR Regression can not be applied"+str(test_ratio), 'danger')
        return jsonify({'error': True}), 400


 