
def fraction(all,poi):
    if all != 'NaN' and poi != 'NaN':
        return float(poi)/all
    return 0


def new_feature(data_dict):

    for key in data_dict.keys():

        emails_sent = data_dict[key]['from_messages']
        to_poi = data_dict[key]['from_this_person_to_poi']
        received_emails = data_dict[key]['to_messages']
        from_poi = data_dict[key]['from_poi_to_this_person']

        data_dict[key]['to_poi_frac'] = fraction(emails_sent, to_poi)
        data_dict[key]['from_poi_frac'] = fraction(received_emails,from_poi)

    return data_dict